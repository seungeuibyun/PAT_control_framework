"""Leased, bounded internal mirror position control through the live broker."""
from __future__ import annotations
import math
from pathlib import Path
import time
import uuid
from shared_acquisition import Broker
from find_pd_maximum import SearchCancelled


class XYMotion:
    def __init__(self, connection=None):
        self.broker = Broker(connection or Path(__file__).with_name('live_gui_connection.json'))
        self.search_id = uuid.uuid4().hex
        self.client = 'receiver-xy-'+self.search_id[:12]
        self.sequence, self.active = 0, False
        self.origin = None

    def state(self):
        s = self.broker.state()
        f = s['fsm']
        if self.active and not f.get('automatic_search_active'):
            raise SearchCancelled('The viewer, watchdog or device cancelled the automatic lease')
        if not f.get('connected') or f.get('fault') or f.get('age_s', 9) > .5:
            raise RuntimeError('Fresh healthy mirror feedback is unavailable')
        if self.active and (f.get('operation_mode') != 2 or f.get('device_status', -1) & ~16):
            raise RuntimeError('Mirror position mode or device status changed')
        if self.origin is not None and math.dist(f['feedback_xy'], self.origin) > .075:
            raise RuntimeError('Actual mirror position exceeded the local calibration bound')
        return s

    def _post(self, action, **values):
        self.sequence += 1
        return self.broker.post('/internal/fsm', dict(action=action, client_id=self.client,
            sequence=self.sequence, search_id=self.search_id, **values))

    def start(self):
        initial = self.state()
        self.origin = list(initial['fsm']['feedback_xy'])
        self.broker.post('/internal/fsm-search', dict(action='begin',
            search_id=self.search_id, client_id=self.client))
        try:
            self._post('xy_mode')
            self.active = True
            time.sleep(.4)
            self.origin = list(self.state()['fsm']['target_xy'])
            self.command(self.origin)
        except Exception:
            self.close()
            raise
        return initial

    def command(self, target):
        if math.dist(target,self.origin) > .07 or not all(math.isfinite(v) for v in target):
            raise ValueError('Position target exceeds the bounded measured region')
        self.state()
        self._post('set_xy', x=float(target[0]), y=float(target[1]))
        deadline, stable, previous = time.monotonic()+15, 0, None
        while time.monotonic() < deadline:
            s = self.state(); f = s['fsm']
            reached = max(abs(a-b) for a,b in zip(f['applied_xy'],target)) < 3e-6
            error = math.dist(f['feedback_xy'],target)
            still = previous is not None and math.dist(f['feedback_xy'],previous) < .00015
            stable = stable+1 if reached and error < .0002 and still else 0
            if stable >= 3:
                time.sleep(.12)
                return s
            previous = f['feedback_xy']; time.sleep(.1)
        raise RuntimeError('Actual mirror position did not follow its bounded XY command')

    def stream_command(self, target):
        """Bounded XY setpoint update without waiting for a stationary mirror."""
        if math.dist(target, self.origin) > .07 or not all(math.isfinite(v) for v in target):
            raise ValueError('Position target exceeds the bounded measured region')
        self.state()
        return self._post('set_xy', x=float(target[0]), y=float(target[1]))

    def close(self):
        try:
            return self.broker.post('/internal/fsm-search',dict(action='end',
                search_id=self.search_id,client_id=self.client))
        finally:
            self.active = False
