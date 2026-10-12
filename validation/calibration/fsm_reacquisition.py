"""Bounded receiver mirror search, gated by actual PD and usable PSD readout.

The search uses receiver mirror coordinates and a prior receiver pose only.
No transmitter position, optical model prediction or synthesized detector
sample determines whether the real beam has been reacquired.
"""
from __future__ import annotations

import math
import time

import numpy as np


class FSMReacquisition:
    def __init__(self, motion, model, dark_voltage, publish=None, timeout=90.):
        self.motion,self.model=motion,model
        self.dark=float(dark_voltage)
        self.threshold=self.dark+.20
        self.publish=publish or (lambda report: None)
        self.timeout=timeout
        self.report={'state':'preflight','search_controller':'bounded_receiver_raster',
            'PD_dark_V':self.dark,'PD_detection_threshold_V':self.threshold,
            'PD_detection_margin_V':.20,'source_stage_position_used':False,
            'laser_or_stage_set_commands_issued':False,'PINN_used_during_search':False,
            'loss_samples':[],'search_samples':[],'PD_seen_count':0}
        self.started=time.monotonic();self.last_publish=0.

    def state(self):
        if time.monotonic()-self.started>self.timeout:
            raise RuntimeError('Receiver search exceeded its finite time limit')
        s=self.motion.state();f=s['fsm'];d=s['pd']
        if f.get('operation_mode')!=2 or f.get('device_status',-1)&~16 \
                or max(map(abs,f.get('actual_current_mA',[float('inf')]*2)))>90:
            raise RuntimeError('Mirror status or current is unsafe for receiver search')
        if not d.get('connected') or d.get('age_s',9)>1.2 \
                or 'DS1ZD190300117' not in d.get('identity','') \
                or not math.isfinite(d.get('mean_voltage_V',float('nan'))):
            raise RuntimeError('Actual PD readout is unavailable during receiver search')
        return s

    def row(self,s):
        p,f,d=s['psd'],s['fsm'],s['pd']
        return {'elapsed_s':time.monotonic()-self.started,'PD_time':d['time'],
            'PD_V':d['mean_voltage_V'],'PSD_time':p.get('time'),
            'PSD_mm':[p.get('x_mm'),p.get('y_mm')],
            'PSD_relative_percent':p.get('relative_percent'),
            'PSD_position_available':p.get('position_available',False),
            'FSM_feedback_xy':f['feedback_xy'],'FSM_target_xy':f['target_xy'],
            'mirror_inside_PINN_support':self.model.contains(f['feedback_xy']) \
                and self.model.contains(f['target_xy'])}

    def notify(self,force=False):
        if force or time.monotonic()-self.last_publish>=1:
            self.report['elapsed_s']=time.monotonic()-self.started
            self.publish(self.report);self.last_publish=time.monotonic()

    def detected(self,s):
        return self.threshold<s['pd']['mean_voltage_V']<4.5

    def position_usable(self,s):
        p,f=s['psd'],s['fsm']
        return bool(p.get('connected') and p.get('age_s',9)<=.5
            and p.get('position_available') and 2<=p.get('relative_percent',-1)<=90
            and p.get('edge_margin_mm',0)>=.5
            and self.model.contains(f['feedback_xy']) and self.model.contains(f['target_xy']))

    def confirm_capture(self,s):
        # Arrest the search at the actual mirror pose, then require independent
        # scope readings while the PSD and measured neural support remain valid.
        self.report['state']='confirming_PD_and_PSD'
        candidate=list(s['fsm']['feedback_xy'])
        self.motion.command(candidate)
        samples=[];last=None;deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            s=self.state()
            if not self.position_usable(s):return False
            if s['pd']['time']!=last:
                last=s['pd']['time']
                if not self.detected(s):return False
                samples.append(self.row(s))
                if len(samples)>=3:
                    self.report.update(state='real_PD_PSD_reacquired',capture_samples=samples,
                        capture_PSD_mm=[s['psd']['x_mm'],s['psd']['y_mm']],
                        capture_FSM_xy=s['fsm']['feedback_xy'],
                        PD_detection_trigger_verified=True,PSD_position_for_PINN_verified=True)
                    self.notify(True);return True
            self.notify();time.sleep(.08)
        return False

    def run(self,anchor,deliberate_loss=True):
        anchor=np.asarray(anchor,float)
        if not self.model.contains(anchor):
            raise RuntimeError('Search anchor must be an actually calibrated receiver pose')
        self.report['anchor_FSM_xy']=anchor.tolist()
        self.report['deliberate_mispoint_requested']=deliberate_loss
        extent=.030
        if deliberate_loss:
            self.report['state']='deliberate_mispoint';self.notify(True)
            self.motion.command((anchor+[extent,0.]).tolist())
            last=None;deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                s=self.state()
                if s['pd']['time']!=last:
                    last=s['pd']['time'];self.report['loss_samples'].append(self.row(s))
                    recent=self.report['loss_samples'][-3:]
                    if len(recent)==3 and all(r['PD_V']<self.dark+.05
                            and not r['PSD_position_available'] for r in recent):
                        # The acquisition service retains its last valid
                        # percent value when the PSD cannot return a position.
                        # It cannot establish fresh optical power in this state.
                        self.report['actual_PD_loss_and_PSD_position_loss_verified']=True;break
                self.notify();time.sleep(.08)
            if not self.report.get('actual_PD_loss_and_PSD_position_loss_verified'):
                raise RuntimeError('The bounded mispoint did not establish PD loss and invalid PSD position')
        self.report['state']='searching';self.notify(True)
        # Start with a horizontal sweep, followed by neighbouring raster rows.
        # 0.0004 receiver XY steps limit the PSD-plane scan speed sufficiently
        # for the independent 0.5-second scope measurements to trigger capture.
        last_pd=None
        for j,y in enumerate((0.,.003,-.003,.006,-.006)):
            xs=np.linspace(extent,-extent,151)
            if j%2:xs=xs[::-1]
            if j:self.motion.command((anchor+[xs[0],y]).tolist())
            for x in xs:
                target=anchor+[x,y]
                if math.dist(target,anchor)>.035:
                    raise RuntimeError('Receiver search exceeded its angular test window')
                self.motion.stream_command(target.tolist())
                deadline=time.monotonic()+.15
                while time.monotonic()<deadline:
                    s=self.state();row=self.row(s)
                    if s['pd']['time']!=last_pd:
                        last_pd=s['pd']['time'];self.report['search_samples'].append(row)
                        if self.detected(s):
                            self.report['PD_seen_count']+=1
                            if self.position_usable(s) and self.confirm_capture(s):
                                return self.report
                            self.report['state']='searching'
                    self.notify();time.sleep(.05)
        raise RuntimeError('No verified PD/PSD capture was found in the bounded receiver scan')
