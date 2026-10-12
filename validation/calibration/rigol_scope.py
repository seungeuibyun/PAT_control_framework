"""Serialized SCPI measurements from the confirmed RIGOL MSO1104Z over LAN."""
from __future__ import annotations

import math
import socket
import threading

from vxi11 import rpc, vxi11


class _BoundConnection:
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self._socket_timeout)
        try:
            if self._local_ip:
                self.sock.bind((self._local_ip, 0))
            self.sock.connect((self.host, self.port))
        except Exception:
            self.sock.close()
            raise


class _PortMapper(_BoundConnection, rpc.TCPPortMapperClient):
    def __init__(self, host, local_ip, timeout):
        self._local_ip, self._socket_timeout = local_ip, timeout
        super().__init__(host)


class _CoreClient(_BoundConnection, vxi11.CoreClient):
    def __init__(self, host, port, local_ip, timeout):
        self._local_ip, self._socket_timeout = local_ip, timeout
        super().__init__(host, port=port)


class RigolScope:
    def __init__(self, ip, local_ip="192.168.10.3", timeout=3.0):
        self.core, self.link = None, None
        self.timeout_ms = max(1, int(timeout * 1000))
        self._lock = threading.RLock()
        self.backend = "VXI-11"
        mapper = _PortMapper(ip, local_ip, timeout + 1)
        try:
            port = mapper.get_port((vxi11.DEVICE_CORE_PROG,
                                    vxi11.DEVICE_CORE_VERS, rpc.IPPROTO_TCP, 0))
        finally:
            mapper.close()
        if not port:
            raise RuntimeError("Oscilloscope VXI-11 service is unavailable")
        try:
            self.core = _CoreClient(ip, port, local_ip, timeout + 1)
            error, self.link, _, self.max_write = self.core.create_link(
                0x50415430, False, self.timeout_ms, b"inst0")
            if error:
                self.link = None
                raise RuntimeError(f"VXI-11 create_link failed: {error}")
            # Clear stale I/O only; preserve all oscilloscope acquisition settings.
            self._check(self.core.device_clear(self.link, 0, self.timeout_ms,
                                               self.timeout_ms), "device_clear")
            self.identity = self.query("*IDN?")
            if not self.identity.startswith("RIGOL TECHNOLOGIES,MSO1104Z,"):
                raise RuntimeError(f"Unexpected oscilloscope: {self.identity!r}")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(error, operation):
        if error:
            raise RuntimeError(f"VXI-11 {operation} failed: {error}")

    def write(self, command):
        with self._lock:
            data = (command + "\n").encode("ascii")
            if len(data) > self.max_write:
                raise ValueError("SCPI command exceeds instrument receive limit")
            error, size = self.core.device_write(self.link, self.timeout_ms,
                                                 self.timeout_ms, vxi11.OP_FLAG_END, data)
            self._check(error, "device_write")
            if size != len(data):
                raise RuntimeError("Incomplete SCPI write")

    def _query_bytes(self, command):
        with self._lock:
            self.write(command)
            chunks, total = [], 0
            while True:
                error, reason, chunk = self.core.device_read(
                    self.link, 65536, self.timeout_ms, self.timeout_ms, 0, 0)
                self._check(error, "device_read")
                if not chunk:
                    raise RuntimeError("Empty oscilloscope reply")
                chunks.append(chunk)
                total += len(chunk)
                if total > 200000:
                    raise RuntimeError("Oscilloscope reply exceeds configured size limit")
                if reason & (vxi11.RX_END | vxi11.RX_CHR):
                    return b"".join(chunks)

    def query(self, command):
        return self._query_bytes(command).decode("ascii").strip()

    def mean_voltage(self):
        value = float(self.query(":MEASure:ITEM? VAVG,CHANnel1"))
        if not math.isfinite(value) or abs(value) >= 1e30:
            raise RuntimeError("CH1 mean-voltage measurement is unavailable")
        return value

    def waveform(self):
        with self._lock:
            preamble = self.query(":WAVeform:PREamble?").split(",")
            block = self._query_bytes(":WAVeform:DATA?")
            marker = block[:2]
            if len(marker) != 2 or marker[:1] != b"#" or not marker[1:2].isdigit():
                raise RuntimeError(f"Invalid waveform block header: {marker!r}")
            digits = int(marker[1:2])
            if digits == 0 or len(block) < 2 + digits:
                raise RuntimeError("Incomplete waveform length header")
            length = int(block[2:2 + digits])
            if not 0 < length <= 100000:
                raise RuntimeError("Unexpected waveform size")
            start = 2 + digits
            payload = block[start:start + length]
            if len(payload) != length or block[start + length:] not in (b"", b"\n", b"\r\n"):
                raise RuntimeError("Incomplete waveform block")
            if len(preamble) != 10 or int(preamble[0]) != 0:
                raise RuntimeError("Waveform reader requires BYTE format")
            increment, origin, reference = map(float, preamble[7:10])
            return {"preamble": preamble, "adc": list(payload),
                    "volts": [(v - origin - reference) * increment for v in payload]}

    def close(self):
        if self.core is not None:
            try:
                if self.link is not None:
                    self._check(self.core.destroy_link(self.link), "destroy_link")
            finally:
                self.core.close()
                self.core, self.link = None, None
