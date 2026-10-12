"""Bounded communication with the verified Optilab source through Termite.

Preserve the user's existing serial session and any pending input draft.
Current changes normally stay at or below the initial verified current.
An explicitly requested alignment increase can opt into a ceiling no more
than 4 mA above that initial setting, also below the read device protections.
Do not change TEC, protection current, maximum current or device address.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
import re
import time


class TermiteLaser:
    def __init__(self, authorised_alignment_ceiling_mA=None):
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.user.GetDlgItem.argtypes = [w.HWND, ctypes.c_int]
        self.user.GetDlgItem.restype = w.HWND
        self.user.SendMessageTimeoutW.argtypes = [w.HWND, w.UINT, w.WPARAM,
            w.LPARAM, w.UINT, w.UINT, ctypes.POINTER(ctypes.c_size_t)]
        self.user.SendMessageTimeoutW.restype = w.LPARAM
        callback = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        windows = []

        @callback
        def collect(hwnd, _):
            title = ctypes.create_unicode_buffer(512)
            self.user.GetWindowTextW(hwnd, title, len(title))
            if title.value.lower().startswith('termite'):
                windows.append(hwnd)
            return True

        self.user.EnumWindows(collect, 0)
        if len(windows) != 1:
            raise RuntimeError('Expected one existing Termite window')
        self.hwnd = windows[0]
        self.input = self.user.GetDlgItem(self.hwnd, 101)
        self.receive = self.user.GetDlgItem(self.hwnd, 102)
        self.send = self.user.GetDlgItem(self.hwnd, 1)
        self.settings = self.user.GetDlgItem(self.hwnd, 100)
        if not all((self.input, self.receive, self.send, self.settings)):
            raise RuntimeError('Expected Termite controls are unavailable')
        settings = self.text(self.settings)
        if not settings.startswith('COM3 9600 bps, 8N1, no handshake'):
            raise RuntimeError('Termite settings do not match the verified laser')
        self.initial = self.read()
        self.upper_current_mA = self.initial['current_mA']
        if authorised_alignment_ceiling_mA is not None:
            ceiling = authorised_alignment_ceiling_mA
            if (isinstance(ceiling, bool) or not isinstance(ceiling, int)
                    or not self.upper_current_mA <= ceiling <= self.upper_current_mA + 4):
                raise ValueError('Alignment increase must be at most 4 mA above the initial current')
            protections = [int(re.search(pattern, self.initial['raw_reply']).group(1))
                           for pattern in (r'Protect Current:\s+(\d+)', r'Abs MAX Current:\s+(\d+)')]
            if ceiling >= min(protections):
                raise ValueError('Alignment ceiling must stay below both source protections')
            self.upper_current_mA = ceiling

    def message(self, hwnd, msg, wp=0, lp=0):
        result = ctypes.c_size_t()
        if not self.user.SendMessageTimeoutW(hwnd, msg, wp, lp, 2, 500,
                                             ctypes.byref(result)):
            raise RuntimeError('Termite window did not respond')
        return result.value

    def text(self, hwnd):
        n = self.message(hwnd, 0xE)
        buf = ctypes.create_unicode_buffer(n + 1)
        self.message(hwnd, 0xD, n + 1, ctypes.addressof(buf))
        return buf.value

    def query(self, command, timeout=3):
        if self.text(self.input).strip():
            raise RuntimeError('Termite contains an unsent user draft; preserved')
        before = self.text(self.receive)
        buffer = ctypes.create_unicode_buffer(command)
        self.message(self.input, 0xC, 0, ctypes.addressof(buffer))
        self.message(self.hwnd, 0x111, 1, self.send)
        deadline, last, stable = time.monotonic() + timeout, '', 0
        while time.monotonic() < deadline:
            time.sleep(.10)
            current = self.text(self.receive)
            delta = current[len(before):] if current.startswith(before) else current[-4000:]
            stable = stable + 1 if delta == last else 0
            last = delta
            if ('Monitor Temp:' in delta or 'Successful' in delta) and stable >= 2:
                return delta
        return last

    def read(self):
        raw = self.query('READ01')
        if ('Model Number: DFB-SM-M-1560-25' not in raw
                or 'Serial Number: H14423462513' not in raw
                or 'Device ID: 01' not in raw):
            raise RuntimeError('Fresh reply does not match the verified laser')
        def value(pattern):
            match = re.search(pattern, raw)
            if match is None:
                raise RuntimeError('Required source report field is missing')
            return float(match.group(1))
        power = re.search(r'Optical Power:\s+([+-]?[\d.]+)', raw)
        return {'current_mA': int(value(r'Set Current:\s+(\d+)')),
                'power_dBm': float(power.group(1)) if power else None,
                'temperature_C': value(r'Monitor Temp:\s+([\d.]+)'),
                'raw_reply': raw}

    def set_current(self, current_mA):
        if (isinstance(current_mA, bool) or not isinstance(current_mA, int)
                or not 0 <= current_mA <= self.upper_current_mA):
            raise ValueError('Current must be an integer within the verified current ceiling')
        before = self.read()
        command = f'SET01C:{current_mA:04d}'
        reply = self.query(command, timeout=.7)
        after = self.read()
        if after['current_mA'] != current_mA:
            raise RuntimeError('Source current readback disagrees with the requested value')
        return {'command': command, 'before': before, 'reply': reply, 'after': after}
