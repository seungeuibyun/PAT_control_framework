"""Serve the legacy PSD position/gauge layout to the monitoring PC over LAN.

Acquisition runs without a desktop window. A single owner serves each serial
device. The configured monitoring PC can also control the physical FSM.
"""
from __future__ import annotations

import argparse
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import math
from pathlib import Path
import re
import secrets
import socket
import threading
import time
import sys

import serial
from serial.tools import list_ports

ROOT = Path(__file__).resolve().parent
CAL = ROOT / "calibration"
sys.path.insert(0, str(CAL))
from rigol_scope import RigolScope
from manual_fsm import ManualFSM


class SingleOwnerHTTPServer(ThreadingHTTPServer):
    """Do not let Windows bind a second acquisition owner to the same port."""
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


def psd_sample_status(x, y, power):
    """Reject absent light, full-scale readings and boundary coordinates."""
    margin = 5 - max(abs(x), abs(y))
    if power <= 0:
        state = "no_signal"
    elif power >= 100:
        state = "over_range"
    elif margin <= 0:
        state = "position_limit"
    else:
        state = "weak" if power <= 1 else "measuring"
    return {"position_available": state in {"weak", "measuring"},
            "signal_state": state, "edge_margin_mm": margin}


class Acquisition:
    def __init__(self, args):
        self.args = args
        self.psd_lock, self.scope_lock, self.state_lock = (threading.RLock() for _ in range(3))
        self.stop = threading.Event()
        self.port, self.scope = None, None
        self.state = {
            "ui_revision": "fsm-manual-2",
            "host": socket.gethostname(), "source_ip": args.bind,
            "requested_monitor_pc": args.viewer_ip,
            "psd": {"port": args.psd_port, "connected": False, "sample_count": 0},
            "fsm": {},
            "pd": {"connected": False, "channel": "CH1", "scope_ip": args.scope_ip},
            "calibration": {"electrical_dark": False, "pd_reference": False, "fsm_mapping": False},
            "viewers": {},
            "legacy_gui": "PAT/PAT/FSM_PSD/conex_pat_modules/simple_gui.py",
        }
        conversion_path = CAL / "PD_readout_conversion.json"
        if conversion_path.exists():
            try:
                conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
                if conversion.get("model") == "PDA015C2":
                    self.state["calibration"]["PD_voltage_conversion"] = conversion
            except (OSError, ValueError, AttributeError) as exc:
                logging.warning("PD display conversion configuration: %s", exc)
        self.fsm = ManualFSM(args.fsm_port, self.update, self.stop)
        self.threads = []

    def update(self, section, **values):
        with self.state_lock:
            self.state[section].update(values)

    def snapshot(self):
        with self.state_lock:
            state = json.loads(json.dumps(self.state))
        state["server_time"] = dt.datetime.now().astimezone().isoformat()
        now = time.time()
        for section in ("psd", "pd", "fsm"):
            state[section]["age_s"] = now - state[section]["time"] if "time" in state[section] else None
        if not state["psd"].get("position_available"):
            state["psd"]["raw_position_mm"] = [state["psd"].get("x_mm"), state["psd"].get("y_mm")]
            state["psd"]["x_mm"] = state["psd"]["y_mm"] = None
        if state["fsm"]["age_s"] is None or state["fsm"]["age_s"] > 2:
            state["fsm"]["control_available"] = False
        return state

    def viewer_seen(self, ip):
        with self.state_lock:
            self.state["viewers"][ip] = dt.datetime.now().astimezone().isoformat()
        self.fsm.heartbeat(ip)

    def psd_command(self, command):
        reads = {"1GP", "1GP?", "1RA?", "1PX?", "1PY?", "1PS?", "1OF?", "1LF?", "1ID?", "1VE?"}
        is_write = command.startswith("1OF") and command != "1OF?"
        if command not in reads:
            if not is_write:
                raise ValueError("Unsupported PSD command")
            values = [float(v) for v in command[3:].split(",")]
            if len(values) != 4 or not all(math.isfinite(v) and abs(v) <= 10 for v in values):
                raise ValueError("Invalid electrical offset values")
        with self.psd_lock:
            if self.port is None:
                raise RuntimeError("PSD is not connected")
            self.port.reset_input_buffer()
            self.port.write((command + "\r\n").encode("ascii"))
            self.port.flush()
            if is_write:
                time.sleep(.08)
                return ""
            raw = self.port.read_until(b"\n").decode("ascii").strip()
            if not raw.startswith(command.rstrip("?")):
                raise RuntimeError(f"Unexpected PSD response: {raw!r}")
            if command.rstrip("?") == "1GP":
                values = [float(v) for v in raw[3:].split(",")]
                if len(values) != 3 or not all(math.isfinite(v) for v in values):
                    raise RuntimeError("Invalid PSD sample")
                x, y, power = values
                with self.state_lock:
                    count = self.state["psd"]["sample_count"] + 1
                self.update("psd", connected=True, x_mm=x, y_mm=y,
                            relative_percent=power, sample_count=count,
                            time=time.time(), error=None,
                            **psd_sample_status(x, y, power))
            return raw

    def pd_voltage(self):
        with self.scope_lock:
            if self.scope is None:
                raise RuntimeError("Oscilloscope is not connected")
            value = self.scope.mean_voltage()
            self.update("pd", connected=True, mean_voltage_V=value,
                        time=time.time(), error=None, identity=self.scope.identity)
            return value

    def scope_command(self, command):
        reads = {"*IDN?", ":CHANnel1:SCALe?", ":CHANnel1:OFFSet?",
                 ":CHANnel1:PROBe?", ":CHANnel1:COUPling?",
                 ":CHANnel1:DISPlay?", ":TRIGger:STATus?",
                 ":WAVeform:PREamble?", ":SYSTem:ERRor?"}
        scale_match = re.fullmatch(r":CHANnel1:SCALe ([0-9.]+)", command)
        with self.scope_lock:
            if self.scope is None:
                raise RuntimeError("Oscilloscope is not connected")
            if command in reads:
                return self.scope.query(command)
            if not scale_match:
                raise ValueError("Unsupported oscilloscope command")
            scale = float(scale_match.group(1))
            if scale not in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10):
                raise ValueError("Unsupported CH1 voltage range")
            self.scope.write(command)
            time.sleep(.4)
            actual = float(self.scope.query(":CHANnel1:SCALe?"))
            self.update("pd", scale_V_per_div=actual)
            if not math.isclose(actual, scale, rel_tol=1e-8):
                raise RuntimeError(f"CH1 range did not change: requested {scale}, actual {actual}")
            return str(actual)

    def _psd_loop(self):
        while not self.stop.is_set():
            try:
                if self.port is None:
                    devices = list(list_ports.comports())
                    selected = next((p for p in devices if p.device.upper() == self.args.psd_port.upper()), None)
                    if selected is None or selected.serial_number not in ("FT9B18QSA", "FT9B18QS"):
                        raise RuntimeError("Configured port does not match the verified PSD USB serial")
                    with self.psd_lock:
                        self.port = serial.Serial(self.args.psd_port, 921600, timeout=.5, write_timeout=.5)
                        identity = self.psd_command("1ID?")
                        if "CONEX-PSD10-GE" not in identity:
                            raise RuntimeError("Connected device is not the verified CONEX-PSD10GE")
                        self.update("psd", identity=identity)
                start = time.monotonic()
                self.psd_command("1GP")
                self.stop.wait(max(0, .05 - (time.monotonic() - start)))
            except Exception as exc:
                self.update("psd", connected=False, error=str(exc))
                logging.warning("PSD: %s", exc)
                with self.psd_lock:
                    if self.port is not None:
                        self.port.close()
                        self.port = None
                self.stop.wait(2)

    def _scope_loop(self):
        while not self.stop.is_set():
            try:
                if self.scope is None:
                    with self.scope_lock:
                        self.scope = RigolScope(self.args.scope_ip, self.args.scope_source_ip)
                        self.update("pd", scale_V_per_div=float(self.scope.query(":CHANnel1:SCALe?")))
                self.pd_voltage()
                self.stop.wait(.5)
            except Exception as exc:
                self.update("pd", connected=False, error=str(exc))
                logging.warning("Scope: %s", exc)
                # Clipping can invalidate VAVG without breaking the connection.
                # Transport errors require a new VXI-11 link after a power cycle.
                if not isinstance(exc, RuntimeError) or "unavailable" not in str(exc):
                    with self.scope_lock:
                        if self.scope is not None:
                            try:
                                self.scope.close()
                            except Exception as close_error:
                                logging.warning("Closing scope transport: %s", close_error)
                            finally:
                                self.scope = None
                self.stop.wait(2)

    def start(self):
        for target in (self._psd_loop, self._scope_loop, self.fsm.run):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self.threads.append(t)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=5)
        self.fsm.close()
        with self.psd_lock:
            if self.port is not None:
                self.port.close()
                self.port = None
        with self.scope_lock:
            if self.scope is not None:
                self.scope.close()
                self.scope = None


def handler_for(acquisition, token):
    control_key = secrets.token_urlsafe(32)
    page_path = ROOT / "live_pat_gui.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, mime="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError):
                pass

        def authorized_viewer(self):
            return self.client_address[0] in {"127.0.0.1", acquisition.args.bind, acquisition.args.viewer_ip}

        def do_GET(self):
            if not self.authorized_viewer():
                self.send(403, {"error": "Viewer address is not configured"})
                return
            if self.path == "/":
                acquisition.viewer_seen(self.client_address[0])
                page = page_path.read_text(encoding="utf-8").replace(
                    "__FSM_CONTROL_KEY__", control_key).encode("utf-8")
                self.send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                acquisition.viewer_seen(self.client_address[0])
                self.send(200, acquisition.snapshot())
            elif self.path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path == "/api/fsm":
                origin = self.headers.get("Origin")
                if not self.authorized_viewer() or not secrets.compare_digest(
                    self.headers.get("X-FSM-Control-Key", ""), control_key
                ) or origin != "http://" + self.headers.get("Host", ""):
                    self.send(403, {"error": "FSM control requires the configured viewer and this page"})
                    return
            elif self.client_address[0] != "127.0.0.1" or not secrets.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token):
                    self.send(403, {"error": "Calibration interface is local only"})
                    return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 2048:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Request must be an object")
                if self.path in {"/api/fsm", "/internal/fsm"}:
                    reply = acquisition.fsm.command(data, self.client_address[0])
                    self.send(202, reply)
                elif self.path == "/internal/fsm-search":
                    reply = acquisition.fsm.search_lease(data, self.client_address[0], acquisition.args.viewer_ip)
                    self.send(200, reply)
                elif self.path == "/internal/psd":
                    reply = acquisition.psd_command(data["command"])
                    self.send(200, {"reply": reply})
                elif self.path == "/internal/pd":
                    self.send(200, {"voltage_V": acquisition.pd_voltage()})
                elif self.path == "/internal/scope":
                    self.send(200, {"reply": acquisition.scope_command(data["command"])})
                elif self.path == "/internal/status":
                    acquisition.update("calibration", **data)
                    self.send(200, {"ok": True})
                elif self.path == "/internal/shutdown":
                    self.send(200, {"ok": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self.send(404, {"error": "Not found"})
            except ValueError as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(503, {"error": str(exc)})

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", default="192.168.0.30")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--viewer-ip", default="192.168.0.36")
    p.add_argument("--psd-port", default="COM4")
    p.add_argument("--fsm-port", default="COM5")
    p.add_argument("--scope-ip", default="192.168.10.2")
    p.add_argument("--scope-source-ip", default="192.168.10.3")
    args = p.parse_args()
    for address in (args.bind, args.viewer_ip):
        ipaddress.IPv4Address(address)
    logging.basicConfig(filename=CAL / "live_psd_server.log", level=logging.INFO,
                        encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s")
    acquisition, token = Acquisition(args), secrets.token_urlsafe(32)
    server = SingleOwnerHTTPServer(("0.0.0.0", args.port), handler_for(acquisition, token))
    server.daemon_threads = True
    connection = {"pid": __import__("os").getpid(), "host": socket.gethostname(),
                  "started": dt.datetime.now().astimezone().isoformat(),
                  "view_url": f"http://{args.bind}:{args.port}/", "viewer_ip": args.viewer_ip,
                  "broker_url": f"http://127.0.0.1:{args.port}", "token": token}
    (CAL / "live_gui_connection.json").write_text(json.dumps(connection, indent=2), encoding="utf-8")
    logging.info("Monitor ready at %s for %s", connection["view_url"], args.viewer_ip)
    acquisition.start()
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        acquisition.close()


if __name__ == "__main__":
    main()
