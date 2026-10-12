"""Single-owner, bounded manual current control for the verified MR-E-3.

The UI displays current commands and independent mirror feedback. It never
interprets a mouse coordinate as a calibrated physical mirror position.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
import threading
import time

import optoMDC
from serial.tools import list_ports


class ManualFSM:
    MAX_CURRENT_MA = 150.0
    MAX_FEEDBACK_RADIUS = .75
    MAX_STEP_A = .015
    HEARTBEAT_TIMEOUT_S = 10.0

    def __init__(self, port, publish, stop_event):
        self.port, self.publish, self.stop = port, publish, stop_event
        self.lock = threading.RLock()
        self.board = None
        self.target = [0., 0.]
        self.target_xy = None
        self.xy_origin = None
        self.xy_saved = None
        self.xy_output_states = None
        self.owner = None
        self.heartbeats = {}
        self.sequences = {}
        self.search_id = None
        self.search_key = None
        self.fault = None
        self.controlled = False
        self.sample_count = 0
        self.log_file = Path(__file__).resolve().parent / "fsm_manual_commands.jsonl"
        self.last = {}
        self._publish(connected=False, control_available=False)

    def _publish(self, **values):
        self.last.update(values)
        self.publish("fsm", **self.last, port=self.port, max_current_mA=self.MAX_CURRENT_MA,
                     target_current_mA=[v * 1000 for v in self.target],
                     control_mode="device_xy_servo" if self.target_xy is not None else "manual_current",
                     target_xy=self.target_xy, fault=self.fault,
                     automatic_search_active=self.search_id is not None)

    def _log(self, event, **values):
        row = {"time": dt.datetime.now().astimezone().isoformat(), "event": event, **values}
        with self.log_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def heartbeat(self, viewer_ip):
        with self.lock:
            self.heartbeats[viewer_ip] = time.monotonic()

    @staticmethod
    def _number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("전류 명령은 유한한 숫자여야 합니다.")
        return float(value)

    def _zero(self, reason):
        if self.search_id:
            self._log("search_cancelled", search_id=self.search_id, reason=reason)
        self.search_id = self.search_key = None
        self.target, self.owner = [0., 0.], None
        if self.board is not None:
            if self.target_xy is not None:
                self._leave_xy(preserve=False)
            if self.board.MiscFeatures.GetOperationMode() != 0:
                raise RuntimeError("제어 모드가 변경되어 전류 명령을 중단했습니다.")
            errors = []
            for channel in self.board.channel:
                try:
                    channel.StaticInput.SetCurrent(0.)
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError("0 mA 복귀 실패: " + "; ".join(errors))
            self.controlled = True
        self._log("zero", reason=reason)
        self._publish(moving=False)

    def _leave_xy(self, preserve=True):
        """Transfer the actual drive current before leaving the inner servo."""
        b = self.board
        current = [b.MiscFeatures.GetADCcurrent(i) for i in (0, 1)] if preserve else [0., 0.]
        if not all(math.isfinite(v) and abs(v) <= .09 for v in current):
            current = [0., 0.]
        # Prevent a partially changed signal path from energizing the coils
        # while switching units/control stages. Clear the retained accumulator
        # before restoring the factory configuration in current mode.
        for i in (0,1):
            b.MiscFeatures.SetOutputState(i,0)
        for channel in b.channel:
            for register in ('kp','ki','kd','min_integral','max_integral'):
                channel.XYPID.set_register(register,0.)
        time.sleep(.05)
        for channel, value in zip(b.channel, current):
            channel.StaticInput.SetCurrent(float(value))
        b.MiscFeatures.SetOperationMode(0)
        if b.MiscFeatures.GetOperationMode() != 0:
            raise RuntimeError("Could not restore current operation mode")
        time.sleep(.08)
        self.target = current
        self.target_xy = self.xy_origin = None
        if self.xy_saved is not None:
            for channel, saved in zip(b.channel, self.xy_saved):
                for register, value in saved['pid'].items():
                    channel.XYPID.set_register(register, value)
                channel.LinearOutput.SetCurrentLimit(saved['positive_current_limit_A'])
                channel.LinearOutput.SetCurrentLimitNegative(saved['negative_current_limit_A'])
            self.xy_saved = None
        if self.xy_output_states is not None:
            for i,enabled in enumerate(self.xy_output_states):
                b.MiscFeatures.SetOutputState(i,enabled)
            self.xy_output_states = None
        self._log("inner_xy_servo_left", held_current_mA=[v*1000 for v in current])

    def _enter_xy(self, diagnostics_only=False):
        """Use the factory XY servo with temporarily reduced current limits.

        This servo uses the mirror's internal feedback. The external PSD PAT
        command is supplied separately by the receiver controller.
        """
        if self.target_xy is not None:
            return
        self._read()
        if self.board.MiscFeatures.GetOperationMode() != 0:
            raise RuntimeError("XY commissioning requires the verified current mode")
        origin = list(self.last['feedback_xy'])
        registers = ('kp', 'ki', 'kd', 'min_integral', 'max_integral')
        self.xy_saved = [{'pid':{r:c.XYPID.get_register(r) for r in registers},
                         'positive_current_limit_A':c.LinearOutput.GetCurrentLimit(),
                         'negative_current_limit_A':c.LinearOutput.GetCurrentLimitNegative()}
                        for c in self.board.channel]
        self.xy_output_states = [self.board.MiscFeatures.GetOutputState(i) for i in (0,1)]
        self.target_xy, self.xy_origin = origin, origin.copy()
        try:
            for i in (0,1):
                self.board.MiscFeatures.SetOutputState(i,0)
            time.sleep(.05)
            for channel, value in zip(self.board.channel, origin):
                channel.StaticInput.SetXY(float(value))
                channel.LinearOutput.SetCurrentLimit(.075)
                channel.LinearOutput.SetCurrentLimitNegative(.075)
                positive, negative = channel.LinearOutput.GetCurrentLimit(), channel.LinearOutput.GetCurrentLimitNegative()
                # Installed firmware reports the negative limit as a positive
                # magnitude, despite the SDK documentation's signed value.
                if abs(positive-.075) > 1e-6 or abs(abs(negative)-.075) > 1e-6:
                    raise RuntimeError(f'Hardware coil-current bounds failed readback: {positive}, {negative}')
                # Commission slow sub-millimetre receiver moves without a
                # derivative kick at mode entry. Original gains are restored
                # on leaving this temporary inner position servo.
                # Clamp the retained integral accumulator to zero before
                # enabling another position loop; it survives mode changes.
                for register, bound in (('kp', 0.), ('ki', 0.), ('kd', 0.),
                                        ('min_integral', 0.), ('max_integral', 0.)):
                    channel.XYPID.set_register(register, bound)
                    actual = channel.XYPID.get_register(register)
                    if abs(actual-bound) > 1e-6:
                        raise RuntimeError(f'Inner-servo {register}: expected {bound}, read {actual}')
            self.board.MiscFeatures.SetOperationMode(2)
            if self.board.MiscFeatures.GetOperationMode() != 2:
                raise RuntimeError("Device did not accept XY position control")
            time.sleep(.05)
            initial_errors = [self.board.get_value({'id':0xC01A+(i<<8),'type':float})[0]
                              for i in (0,1)]
            self._log('inner_xy_zero_gain_readback', position_errors_xy=initial_errors,
                      actual_feedback_xy=[self.board.RadialBasisFunction.GetMirrorCoordinateX(),
                                          self.board.RadialBasisFunction.GetMirrorCoordinateY()],
                      actual_current_A=[self.board.MiscFeatures.GetADCcurrent(i) for i in (0,1)])
            if diagnostics_only:
                diagnostics = {'held_target_xy':origin, 'samples':[],
                               'gains_are_zero':True, 'motion_command_issued':False}
                try:
                    sfm = self.board.SignalFlowManager
                    diagnostics['main_control_systems'] = [sfm.GetStageSystem(0,2,i) for i in (0,1)]
                    diagnostics['feedback_output_systems'] = [sfm.GetStageSystem(2,4,i) for i in (0,1)]
                    for _ in range(3):
                        sample = {'feedback_xy':[self.board.RadialBasisFunction.GetMirrorCoordinateX(),
                                                  self.board.RadialBasisFunction.GetMirrorCoordinateY()],
                                  'position_error_xy':[self.board.get_value({'id':0xC01A+(i<<8),'type':float})[0] for i in (0,1)],
                                  'main_input':[sfm.GetStageOutput(0,0,i) for i in (0,1)],
                                  'main_conditioned_input':[sfm.GetStageOutput(0,1,i) for i in (0,1)],
                                  'feedback_output':[sfm.GetStageOutput(2,4,i) for i in (0,1)],
                                  'actual_current_A':[self.board.MiscFeatures.GetADCcurrent(i) for i in (0,1)],
                                  'output_current_A':[c.LinearOutput.GetOutputCurrent() for c in self.board.channel],
                                  'status':self.board.Status.GetDeviceStatus()}
                        diagnostics['samples'].append(sample)
                        if max(map(abs,sample['actual_current_A'])) > .004 or sample['status'] & ~16:
                            raise RuntimeError('Zero-gain diagnostics encountered nonzero current or a device fault')
                        time.sleep(.08)
                    self._log('zero_gain_xy_diagnostics', **diagnostics)
                    return diagnostics
                finally:
                    self._leave_xy(preserve=False)
            # Entering position mode can release a remanent zero-current
            # pose. Capture the settled actual pose in the new signal flow
            # before enabling any gain, instead of chasing the old pose.
            settled = []
            for _ in range(4):
                settled.append([self.board.RadialBasisFunction.GetMirrorCoordinateX(),
                                self.board.RadialBasisFunction.GetMirrorCoordinateY()])
                time.sleep(.08)
            if math.dist(settled[-1],settled[-2]) > .00015 or math.dist(settled[-1],origin) > .02:
                raise RuntimeError('Mirror did not settle within the bounded zero-gain mode transfer')
            origin = settled[-1]
            self.target_xy, self.xy_origin = origin.copy(),origin.copy()
            for channel,value in zip(self.board.channel,origin):
                channel.StaticInput.SetXY(float(value))
            time.sleep(.04)
            transferred_errors = [self.board.get_value({'id':0xC01A+(i<<8),'type':float})[0] for i in (0,1)]
            if max(map(abs,transferred_errors)) > .0003:
                raise RuntimeError('Bumpless mode transfer position error remains excessive')
            self._log('inner_xy_bumpless_transfer', settled_target_xy=origin,
                      position_error_xy=transferred_errors)
            for channel,saved in zip(self.board.channel,self.xy_saved):
                # Integral bounds are accumulator units, not coil amperes.
                # Restore their manufacturer values after clearing the state;
                # actual current remains bounded by both output stages.
                for register, value in (('min_integral',saved['pid']['min_integral']),
                                        ('max_integral',saved['pid']['max_integral']),
                                        ('kp',2.), ('ki',20.)):
                    channel.XYPID.set_register(register,value)
                    actual=channel.XYPID.get_register(register)
                    if not math.isclose(actual,value,rel_tol=1e-6,abs_tol=1e-6):
                        raise RuntimeError('Soft inner-servo gains failed readback')
            for i,enabled in enumerate(self.xy_output_states):
                self.board.MiscFeatures.SetOutputState(i,enabled)
            self._log('inner_xy_configuration_readback', channels=[{
                'gains':{r:c.XYPID.get_register(r) for r in ('kp','ki','kd')},
                'static_xy':c.StaticInput.GetXY(),
                'active_input_unit':self.board.get_value({'id':c.StaticInput.sys_id<<8|3,'type':int})[0],
                'current_limit_magnitude_A':c.LinearOutput.GetCurrentLimit(),
                'negative_limit_magnitude_A':c.LinearOutput.GetCurrentLimitNegative()}
                for c in self.board.channel])
        except Exception:
            self._leave_xy(preserve=False)
            raise
        self.controlled = True
        self._log("inner_xy_servo_entered", target_xy=origin,
                  hardware_coil_current_limit_A=.075, original_limits=self.xy_saved)
        self._publish()

    def search_lease(self, data, viewer_ip, handoff_viewer):
        """Serialize a local measurement search with browser control and Stop."""
        with self.lock:
            if viewer_ip != "127.0.0.1":
                raise ValueError("Automatic search is local only")
            search_id, client = data.get("search_id"), data.get("client_id")
            if not all(isinstance(v, str) and 1 <= len(v) <= 80 for v in (search_id, client)):
                raise ValueError("Invalid search identity")
            key = (viewer_ip, client)
            if data.get("action") == "begin":
                if self.search_id:
                    raise RuntimeError("An automatic search is already running")
                if self.board is None or self.fault or not self.last.get("control_available"):
                    raise RuntimeError(self.fault or "FSM is not ready for a search")
                self._read()
                self.search_id, self.search_key = search_id, key
                self.owner = viewer_ip
                self.heartbeats[viewer_ip] = time.monotonic()
                self._log("search_started", search_id=search_id)
                self._publish()
                return {"accepted": True}
            if data.get("action") != "end":
                raise ValueError("Unsupported search action")
            if self.search_id != search_id or self.search_key != key:
                return {"accepted": False, "cancelled": True}
            self.search_id = self.search_key = None
            if time.monotonic() - self.heartbeats.get(handoff_viewer, 0) <= self.HEARTBEAT_TIMEOUT_S:
                self.owner = handoff_viewer
            else:
                self._zero("search_finished_without_monitor")
            self._log("search_finished", search_id=search_id, handoff_viewer=handoff_viewer)
            self._publish()
            return {"accepted": True, "target_current_mA": [v * 1000 for v in self.target],
                    "monitor_ip": self.owner}

    def command(self, data, viewer_ip):
        with self.lock:
            if self.board is None or not self.last.get("connected"):
                raise RuntimeError("FSM 연결을 기다려 주세요.")
            action = data.get("action")
            if action not in {"set", "nudge", "zero", "stop", "resume", "xy_mode", "set_xy", "current_mode", "xy_diagnostics"}:
                raise ValueError("지원하지 않는 FSM 동작입니다.")
            client, sequence = data.get("client_id"), data.get("sequence")
            if not isinstance(client, str) or not 1 <= len(client) <= 80:
                raise ValueError("잘못된 제어 세션입니다.")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < 2**53:
                raise ValueError("잘못된 명령 순서입니다.")
            key = (viewer_ip, client)
            if sequence <= self.sequences.get(key, -1):
                raise ValueError("이전 명령은 적용하지 않습니다.")
            if "search_id" in data:
                if self.search_id != data["search_id"] or self.search_key != key:
                    raise RuntimeError("Automatic search was cancelled; no further motion is permitted")
            elif self.search_id and action != "stop":
                raise RuntimeError("PD 최대점 자동 탐색 중입니다. 정지 버튼으로 탐색을 중단할 수 있습니다.")
            if action in {"stop", "resume"}:
                self.sequences[key] = sequence
                self._zero("user_" + action)
                self._read()
                if action == "resume":
                    self.fault = None
                    self._publish(control_available=True)
                return {"accepted": True, "target_current_mA": [0., 0.]}
            if self.fault or not self.last.get("control_available"):
                raise RuntimeError(self.fault or "현재 FSM을 수동 제어할 수 없습니다.")
            if time.time() - self.last.get("time", 0) > 2:
                raise RuntimeError("FSM 위치 피드백이 오래되어 명령을 중단했습니다.")
            if action in {"xy_mode", "set_xy", "current_mode", "xy_diagnostics"}:
                if viewer_ip != "127.0.0.1" or self.search_id is None or data.get("search_id") != self.search_id:
                    raise ValueError("Automatic XY commands require a local active lease")
                self.sequences[key] = sequence
                if action == 'xy_diagnostics':
                    if max(map(abs,self.target)) > 1e-8:
                        raise RuntimeError('Zero-gain diagnostics require idle zero current')
                    diagnostics = self._enter_xy(diagnostics_only=True)
                    self._read()
                    return {'accepted':True,'diagnostics':diagnostics}
                if action == "xy_mode":
                    self._enter_xy()
                elif action == "current_mode":
                    if self.target_xy is not None:
                        self._leave_xy()
                else:
                    if self.target_xy is None:
                        raise RuntimeError("Internal XY servo is not active")
                    target = [self._number(data.get("x")), self._number(data.get("y"))]
                    if math.dist(target, self.xy_origin) > .08 or math.hypot(*target) > .25:
                        raise ValueError("XY command exceeds the bounded local receiver region")
                    self.target_xy = target
                    self._log("xy_command", target_xy=target)
                self.owner = viewer_ip
                self.heartbeats[viewer_ip] = time.monotonic()
                self._publish(moving=True)
                return {"accepted": True, "target_xy": self.target_xy,
                        "target_current_mA": [v*1000 for v in self.target]}
            if self.target_xy is not None:
                self._leave_xy()
            if action == "zero":
                target = [0., 0.]
            elif action == "set":
                target = [self._number(data.get("x_mA")), self._number(data.get("y_mA"))]
            else:
                axis = data.get("axis")
                if axis not in {"x", "y"}:
                    raise ValueError("축은 x 또는 y여야 합니다.")
                target = [v * 1000 for v in self.target]
                target[0 if axis == "x" else 1] += self._number(data.get("delta_mA"))
            if math.hypot(*target) > self.MAX_CURRENT_MA + 1e-8:
                raise ValueError("수동 전류 범위는 두 축 합성 기준 ±150 mA 이내입니다.")
            self.sequences[key] = sequence
            self.target = [v / 1000 for v in target]
            self.owner = viewer_ip
            self.heartbeats[viewer_ip] = time.monotonic()
            self.controlled = True
            self._log("command", action=action, viewer_ip=viewer_ip, target_current_mA=target)
            self._publish(moving=True)
            return {"accepted": True, "target_current_mA": target}

    def _read(self):
        b = self.board
        xy = [b.RadialBasisFunction.GetMirrorCoordinateX(), b.RadialBasisFunction.GetMirrorCoordinateY()]
        current = [b.MiscFeatures.GetADCcurrent(i) for i in (0, 1)]
        setpoint = [c.StaticInput.GetCurrent() for c in b.channel]
        status = b.Status.GetDeviceStatus()
        mode = b.MiscFeatures.GetOperationMode()
        xy_setpoint = [c.StaticInput.GetXY() for c in b.channel] if self.target_xy is not None else None
        temperature = b.TemperatureManager.get_register("device_temperature")
        if not all(math.isfinite(v) for v in xy + current + setpoint + [temperature]):
            raise RuntimeError("FSM의 실제 측정값이 유효하지 않습니다.")
        self.sample_count += 1
        self._publish(connected=True, time=time.time(), sample_count=self.sample_count,
                      feedback_xy=xy, actual_current_mA=[v * 1000 for v in current],
                      applied_current_mA=[v * 1000 for v in setpoint],
                      applied_xy=xy_setpoint,
                      device_status=status, operation_mode=mode, temperature_C=temperature,
                      moving=any(abs(a-v) > 1e-7 for a,v in zip(
                          xy_setpoint if xy_setpoint is not None else setpoint,
                          self.target_xy if self.target_xy is not None else self.target)),
                      error=None, control_available=not self.fault and mode == (2 if self.target_xy is not None else 0))
        # Manufacturer status bit 4 denotes a mirror in motion. Every other
        # status bit, including output-limit history, prevents further motion.
        if mode != (2 if self.target_xy is not None else 0):
            raise RuntimeError("FSM이 검증된 전류 제어 모드가 아닙니다.")
        if status & ~(1 << 4):
            raise RuntimeError(f"FSM 드라이버 상태 {status:#x}: 전류 명령 중단")
        if math.hypot(*xy) > self.MAX_FEEDBACK_RADIUS:
            raise RuntimeError("FSM 실제 위치가 수동 동작 범위를 넘어 전류를 0으로 복귀했습니다.")
        if max(map(abs, current)) > .20 or temperature > 70:
            raise RuntimeError("FSM 전류 또는 온도 범위를 넘어 전류를 0으로 복귀했습니다.")
        if self.target_xy is not None and (max(map(abs,current)) > .09 or math.dist(xy,self.xy_origin) > .085):
            raise RuntimeError("Actual inner-servo current or mirror position exceeded its local bound")
        return setpoint

    def _connect(self):
        selected = next((p for p in list_ports.comports() if p.device.upper() == self.port.upper()), None)
        if selected is None or selected.serial_number != "355933803233":
            raise RuntimeError("COM 포트가 확인된 Optotune FSM과 일치하지 않습니다.")
        self.board = optoMDC.connectmre3(port=self.port, verbose=False)
        self.board.Connection._serial_conn.timeout = .4
        self.board.Connection._serial_conn.write_timeout = .4
        if int(self.board.Status.GetDeviceID(0)) != 25:
            raise RuntimeError("연결된 미러가 확인된 MR-15-30이 아닙니다.")
        if self.controlled and self.fault:
            self._zero("recovery_after_connection_error")
        self.target = [c.StaticInput.GetCurrent() for c in self.board.channel]
        if math.hypot(*self.target) > .02:
            raise RuntimeError("기존 구동 전류가 남아 있어 수동 제어 연결을 중단했습니다.")
        if self.board.MiscFeatures.GetFEpower() != 1 or any(self.board.MiscFeatures.GetOutputState(i) != 1 for i in (0, 1)):
            raise RuntimeError("FSM 구동 전원이 꺼져 있습니다.")
        self._read()
        self._log("connected", initial=self.last)

    def run(self):
        while not self.stop.is_set():
            try:
                with self.lock:
                    if self.board is None:
                        self._connect()
                    setpoint = self._read()
                    if self.fault:
                        self._publish(control_available=False)
                    else:
                        if self.owner and time.monotonic() - self.heartbeats.get(self.owner, 0) > self.HEARTBEAT_TIMEOUT_S:
                            self._zero("viewer_connection_lost")
                            setpoint = [0., 0.]
                        active_target = self.target_xy if self.target_xy is not None else self.target
                        active_setpoint = self.last['applied_xy'] if self.target_xy is not None else setpoint
                        delta = [t-c for t,c in zip(active_target,active_setpoint)]
                        length = math.hypot(*delta)
                        if length > 1e-7:
                            maximum_step = .00075 if self.target_xy is not None else self.MAX_STEP_A
                            fraction = min(1., maximum_step / length)
                            for channel, c, d in zip(self.board.channel, active_setpoint, delta):
                                setter = channel.StaticInput.SetXY if self.target_xy is not None else channel.StaticInput.SetCurrent
                                setter(float(c+fraction*d))
                self.stop.wait(.10)
            except Exception as exc:
                logging.warning("FSM: %s", exc)
                with self.lock:
                    if self.board is not None:
                        self.fault = str(exc)
                        try:
                            if self.controlled:
                                self._zero("fault")
                        except Exception as zero_error:
                            logging.error("FSM zero after fault: %s", zero_error)
                        self._log("fault", error=str(exc))
                        self.board.disconnect()
                        self.board = None
                    self._publish(connected=False, control_available=False, error=str(exc), moving=False)
                self.stop.wait(2.)

    def close(self):
        with self.lock:
            if self.board is not None:
                try:
                    if self.controlled:
                        self._zero("server_shutdown")
                finally:
                    self.board.disconnect()
                    self.board = None
            self._publish(connected=False, control_available=False)
