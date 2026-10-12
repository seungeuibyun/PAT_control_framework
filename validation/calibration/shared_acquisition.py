"""Local client for the single-owner PSD/oscilloscope acquisition service."""
from __future__ import annotations

import json
import math
from pathlib import Path
import urllib.request
import urllib.error


class Broker:
    def __init__(self, state_file):
        self.connection = json.loads(Path(state_file).read_text(encoding="utf-8"))

    def post(self, path, values):
        request = urllib.request.Request(
            self.connection["broker_url"] + path,
            json.dumps(values).encode("utf-8"),
            {"Content-Type": "application/json", "Authorization": "Bearer " + self.connection["token"]},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=6) as reply:
                data = json.load(reply)
        except urllib.error.HTTPError as exc:
            raw = exc.read(4096).decode('utf-8', errors='replace')
            try:
                message = json.loads(raw).get('error', raw)
            except json.JSONDecodeError:
                message = raw
            raise RuntimeError(f'Acquisition broker {path}: {message}') from exc
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    def state(self):
        with urllib.request.urlopen(self.connection["broker_url"] + "/api/state", timeout=3) as reply:
            return json.load(reply)


class BrokerSerial:
    def __init__(self, broker, *_args, **_kwargs):
        self.broker, self.pending = broker, None

    def reset_input_buffer(self):
        self.pending = None

    def write(self, data):
        command = data.decode("ascii").strip()
        if command.startswith("1OF") and command != "1OF?":
            self.broker.post("/internal/psd", {"command": command})
        else:
            self.pending = command
        return len(data)

    def flush(self):
        pass

    def read_until(self, *_args, **_kwargs):
        if self.pending is None:
            raise RuntimeError("PSD read without a pending query")
        command, self.pending = self.pending, None
        return (self.broker.post("/internal/psd", {"command": command})["reply"] + "\n").encode("ascii")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.pending = None


class BrokerScope:
    def __init__(self, broker, *_args, **_kwargs):
        self.broker = broker
        state = broker.state()
        self.identity = state["pd"].get("identity")
        if not self.identity or not self.identity.startswith("RIGOL TECHNOLOGIES,MSO1104Z,"):
            raise RuntimeError("Verified oscilloscope is not available in the acquisition service")

    def mean_voltage(self):
        value = float(self.broker.post("/internal/pd", {})["voltage_V"])
        if not math.isfinite(value) or abs(value) >= 1e30:
            raise RuntimeError("Invalid PD voltage")
        return value

    def close(self):
        pass
