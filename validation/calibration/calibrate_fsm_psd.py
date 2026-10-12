"""Measure a bounded local FSM/PSD response without changing servo or PSD settings.

The current-mode probe is intentional: it preserves the MR-E-3's existing
operation mode. Its command Jacobian is in mm/A, not mm/rad or mm/XY-command.
The independent mirror-feedback fit uses the device's normalized XY readback.
No laser or source-stage commands are issued. Original currents are restored.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import serial


def psd_query(port, command):
    port.reset_input_buffer()
    port.write((command + "\r\n").encode("ascii"))
    port.flush()
    raw = port.read_until(b"\n").decode("ascii", errors="strict").strip()
    if not raw.startswith(command.rstrip("?")):
        raise RuntimeError(f"Unexpected PSD reply to {command}: {raw!r}")
    return raw


def read_psd(port):
    raw = psd_query(port, "1GP")
    values = np.array([float(x) for x in raw[3:].split(",")])
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"Invalid PSD sample: {raw!r}")
    return values


def mirror_xy(board):
    return np.array([board.RadialBasisFunction.GetMirrorCoordinateX(),
                     board.RadialBasisFunction.GetMirrorCoordinateY()], dtype=float)


def acquire_block(board, psd, label, command, reference_power=None,
                  reference_centroid=None, scope=None):
    time.sleep(0.35)
    rows = []
    pd_rows = []
    xy_start = mirror_xy(board)
    for sample_index in range(12):
        sample = read_psd(psd)
        if sample[2] <= 0 or sample[2] >= 95 or np.max(np.abs(sample[:2])) >= 4.25:
            raise RuntimeError(f"PSD signal/edge guard failed: {sample.tolist()}")
        if reference_power is not None and sample[2] < reference_power * 0.5:
            raise RuntimeError("PSD power dropped below half the starting value")
        if reference_centroid is not None and np.linalg.norm(sample[:2] - reference_centroid) > 0.4:
            raise RuntimeError("PSD motion exceeded the 0.4 mm probe guard")
        rows.append([time.monotonic(), *sample.tolist()])
        if scope is not None and sample_index % 3 == 0:
            pd_rows.append([time.monotonic(), scope.mean_voltage()])
        time.sleep(0.025)
    xy_end = mirror_xy(board)
    data = np.array(rows)
    block = {
        "label": label,
        "command_A": np.asarray(command).tolist(),
        "time_monotonic": float(data[:, 0].mean()),
        "centroid_mm": data[:, 1:3].mean(axis=0).tolist(),
        "centroid_std_mm": data[:, 1:3].std(axis=0, ddof=1).tolist(),
        "psd_relative_power_percent": float(data[:, 3].mean()),
        "mirror_feedback_xy": ((xy_start + xy_end) / 2).tolist(),
        "mirror_feedback_start": xy_start.tolist(),
        "mirror_feedback_end": xy_end.tolist(),
        "samples": rows,
    }
    if pd_rows:
        pd_values = np.array(pd_rows)[:, 1]
        block["pd_CH1_mean_voltage_V"] = float(pd_values.mean())
        block["pd_CH1_voltage_std_V"] = float(pd_values.std(ddof=1))
        block["pd_CH1_samples"] = pd_rows
    return block


def fit_response(blocks):
    """Subtract interpolated neighboring baselines to reduce slow thermal drift."""
    delta_current, delta_feedback, delta_psd, noise = [], [], [], []
    for index in (1, 3, 5, 7):
        before, probe, after = blocks[index - 1:index + 2]
        weight = ((probe["time_monotonic"] - before["time_monotonic"]) /
                  (after["time_monotonic"] - before["time_monotonic"]))
        def residual(key):
            baseline = ((1 - weight) * np.array(before[key]) +
                        weight * np.array(after[key]))
            return np.array(probe[key]) - baseline
        delta_current.append(residual("command_A"))
        delta_feedback.append(residual("mirror_feedback_xy"))
        delta_psd.append(residual("centroid_mm"))
        noise.append(np.linalg.norm(probe["centroid_std_mm"]))
    dc, df, dp = map(np.array, (delta_current, delta_feedback, delta_psd))
    current_j = np.linalg.lstsq(dc, dp, rcond=None)[0].T
    feedback_j = np.linalg.lstsq(df, dp, rcond=None)[0].T
    fitted = dc @ current_j.T
    rms = float(np.sqrt(np.mean((fitted - dp) ** 2)))
    feedback_rms = float(np.sqrt(np.mean((df @ feedback_j.T - dp) ** 2)))
    snr = [float(np.linalg.norm(p) / max(n, 0.001)) for p, n in zip(dp, noise)]
    condition = float(np.linalg.cond(current_j))
    feedback_condition = float(np.linalg.cond(feedback_j))
    baseline_drift = float(np.linalg.norm(np.array(blocks[-1]["centroid_mm"]) -
                                          np.array(blocks[0]["centroid_mm"])))
    passed = bool(np.linalg.matrix_rank(dc) == 2 and
                  np.linalg.matrix_rank(df) == 2 and
                  np.isfinite(condition) and condition < 50 and
                  np.isfinite(feedback_condition) and feedback_condition < 50 and
                  min(snr) >= 5 and rms < 0.03 and feedback_rms < 0.03 and baseline_drift < 0.05)
    return {
        "passed": passed,
        "command_jacobian_mm_per_A": current_j.tolist(),
        "observed_feedback_jacobian_mm_per_normalized_xy": feedback_j.tolist(),
        "command_jacobian_condition": condition if np.isfinite(condition) else None,
        "feedback_jacobian_condition": feedback_condition if np.isfinite(feedback_condition) else None,
        "command_fit_rms_mm": rms,
        "feedback_fit_rms_mm": feedback_rms,
        "probe_signal_to_noise": snr,
        "baseline_return_drift_mm": baseline_drift,
        "interpolated_baseline_corrected_psd_shifts_mm": dp.tolist(),
        "interpolated_baseline_corrected_feedback_shifts": df.tolist(),
        "scope": "Local response only. No electrical dark calibration, radian conversion, or PD-optimal reference is implied.",
    }


def run(args):
    step = args.step_A
    if not np.isfinite(step) or not 0 < step <= 0.001:
        raise ValueError("Probe step must be in (0, 0.001] A")
    offsets = np.array([[0, 0], [step, 0], [0, 0], [-step, 0],
                        [0, 0], [0, step], [0, 0], [0, -step], [0, 0]])
    if args.dry_run:
        print(json.dumps({"hardware_opened": False, "probe_offsets_A": offsets.tolist()}, indent=2))
        return
    import optoMDC
    from rigol_scope import RigolScope
    broker = None
    psd_factory, scope_factory = serial.Serial, RigolScope
    if args.broker_state:
        from shared_acquisition import Broker, BrokerSerial, BrokerScope
        broker = Broker(args.broker_state)
        psd_factory = lambda *a, **kw: BrokerSerial(broker, *a, **kw)
        scope_factory = lambda *a, **kw: BrokerScope(broker, *a, **kw)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(__file__).resolve().parent / (f"fsm_psd_current_probe_{stamp}.json")
    result = {"started": dt.datetime.now().astimezone().isoformat(),
              "state": "in_progress", "step_A": step, "blocks": [],
              "hardware": {"fsm_port": args.fsm_port, "psd_port": args.psd_port},
              "source_stage_position_used": False,
              "acquisition": "shared_GUI_service" if broker else "direct_serial",
              "pd_reference_calibrated": False,
              "electrical_dark_offsets_calibrated": False}
    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    scope, board = None, None
    original = None
    motion_started = False
    try:
        if broker:
            broker.post("/internal/status", {"activity": "FSM 응답 측정 중"})
        scope = scope_factory(args.scope_ip, args.scope_source_ip) if args.scope_ip else None
        if scope:
            result["scope_identity"] = scope.identity
            scope.mean_voltage()
        board = optoMDC.connectmre3(port=args.fsm_port, verbose=False)
        if board.MiscFeatures.GetOperationMode() != 0 or board.Status.GetDeviceStatus() != 0:
            raise RuntimeError("Probe requires current operation mode and zero device status errors")
        for i, channel in enumerate(board.channel):
            if channel.Manager.get_register("input") != 0x50 + i:
                raise RuntimeError("Existing input is not StaticInput; no automatic mode changes allowed")
            if channel.Manager.get_register("control") != 0xB0 + i:
                raise RuntimeError("Existing control is not current feed-through")
            if channel.InputConditioning.GetGain() != 1 or channel.InputConditioning.GetOffset() != 0:
                raise RuntimeError("Probe requires unchanged unity input conditioning")
        original = np.array([c.StaticInput.GetCurrent() for c in board.channel])
        if not np.all(np.isfinite(original)) or np.max(np.abs(original)) > 0.01:
            raise RuntimeError("Starting currents exceed the conservative probe window")
        result["original_currents_A"] = original.tolist()
        save()
        with psd_factory(args.psd_port, 921600, timeout=0.7, write_timeout=1) as psd:
            result["psd_settings"] = {q: psd_query(psd, q) for q in
                                      ("1PX?", "1PY?", "1PS?", "1OF?", "1LF?")}
            for i, offset in enumerate(offsets):
                command = original + offset
                if i:
                    motion_started = True
                    for channel, value in zip(board.channel, command):
                        channel.StaticInput.SetCurrent(float(value))
                first = result["blocks"][0] if result["blocks"] else None
                block = acquire_block(board, psd, f"block_{i}", command,
                                      first["psd_relative_power_percent"] if first else None,
                                      np.array(first["centroid_mm"]) if first else None,
                                      scope=scope)
                result["blocks"].append(block)
                save()
                print(json.dumps({"block": i, "current_A": command.tolist(),
                                  "centroid_mm": block["centroid_mm"],
                                  "power_percent": block["psd_relative_power_percent"],
                                  "pd_mean_voltage_V": block.get("pd_CH1_mean_voltage_V")}), flush=True)
            result["fit"] = fit_response(result["blocks"])
            result["state"] = "local_current_response_verified" if result["fit"]["passed"] else "measurement_requires_review"
    except Exception as exc:
        result["state"] = "aborted"
        result["error"] = str(exc)
        raise
    finally:
        restore_errors = []
        if motion_started and original is not None:
            for i, (channel, value) in enumerate(zip(board.channel, original)):
                try:
                    channel.StaticInput.SetCurrent(float(value))
                except Exception as exc:
                    restore_errors.append({"axis": i, "error": str(exc)})
            time.sleep(0.2)
        try:
            if board is not None:
                result["final_currents_A"] = [c.StaticInput.GetCurrent() for c in board.channel]
                result["final_operation_mode"] = board.MiscFeatures.GetOperationMode()
                result["restore_errors"] = restore_errors
                result["restored_original_currents"] = bool(original is not None and
                    not restore_errors and np.allclose(result["final_currents_A"], original, atol=1e-8, rtol=0))
        finally:
            try:
                if board is not None:
                    board.disconnect()
            finally:
                try:
                    if scope is not None:
                        scope.close()
                finally:
                    result["motion_commands_issued"] = motion_started
                    result["finished"] = dt.datetime.now().astimezone().isoformat()
                    save()
                    if broker:
                        try:
                            broker.post("/internal/status", {"activity": "",
                                "fsm_mapping": bool(result.get("fit", {}).get("passed")),
                                "last_probe_state": result["state"]})
                        except Exception as exc:
                            result["GUI_status_update_error"] = str(exc)
                            save()
                    print(json.dumps({"saved": str(output), "state": result["state"],
                                      "restored": result.get("restored_original_currents"),
                                      "fit": result.get("fit")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsm-port", default="COM5")
    parser.add_argument("--psd-port", default="COM4")
    parser.add_argument("--step-A", type=float, default=0.00025)
    parser.add_argument("--scope-ip")
    parser.add_argument("--scope-source-ip", default="192.168.10.3")
    parser.add_argument("--broker-state", help="Use the live GUI acquisition service instead of opening COM4")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())
