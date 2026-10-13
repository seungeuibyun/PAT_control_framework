"""Controllers with the same information: PSD frames, own commands, nominal models.

Hold          FSM fixed at the calibration point (no control).
PID           PSD error -> position PID, updated per 50 Hz frame, held between frames.
Predictive    pseudo-open-loop disturbance d = PSD/f - C_F,nominal * LPF(FSM_nominal(commands)),
              context model predicts d over the next frames, Lanczos interpolation gives
              d at the time the new command takes effect, and the command is the one-shot
              inverse of the nominal plant: theta = C_F^-1 (target/f - d).
Oracle        knows the true disturbance, offset and C_F (bound only; jitter still unknown).
"""
import numpy as np

from pat.predictor import OnlineReadout
from pat.plant import fsm_discretization


class HoldController:
    name = 'hold'

    def reset(self, context):
        self.command = np.zeros((context.batch, 2))

    def on_frame(self, reading, capture_step, now):
        pass

    def tick(self, now):
        return self.command


class PIDController:
    """Position PID on the PSD error; gains may be per-batch arrays for tuning sweeps."""
    name = 'pid'

    def __init__(self, kp, ki, kd, derivative_filter):
        self.kp, self.ki, self.kd = (np.atleast_1d(np.asarray(v, dtype=float))[:, None] for v in (kp, ki, kd))
        self.tf = derivative_filter

    def reset(self, context):
        spec = context.spec
        self.inverse = np.linalg.inv(spec.psd_lever*context.gain_model)
        self.target = context.target
        self.dt = spec.psd.frame_period_s
        self.range = spec.fsm.software_range_rad
        b = context.batch
        self.integral, self.derivative = np.zeros((b, 2)), np.zeros((b, 2))
        self.last_error = None
        self.command = np.zeros((b, 2))

    def on_frame(self, reading, capture_step, now):
        error = np.einsum('bij,bj->bi', self.inverse, self.target-reading)
        if self.last_error is not None:
            raw = (error-self.last_error)/self.dt
            self.derivative += self.dt/(self.tf+self.dt)*(raw-self.derivative)
        self.last_error = error
        proposed = self.integral+self.ki*self.dt*error
        raw_command = self.kp*error+proposed+self.kd*self.derivative
        bounded = np.clip(raw_command, -self.range, self.range)
        # Conditional integration: freeze the integral only when it pushes further out of range.
        self.integral = np.where((raw_command-bounded)*error > 0, self.integral, proposed)
        self.command = np.clip(self.kp*error+self.integral+self.kd*self.derivative, -self.range, self.range)

    def tick(self, now):
        return self.command


class NominalPlant:
    """Nominal MR-15-30 response + PSD filter driven by the controller's own commands."""
    def __init__(self, spec, batch, gain_model, buffer=1024):
        self.ad, self.bd = fsm_discretization(spec)
        self.gain = gain_model
        self.lpf_gain = 1-np.exp(-spec.dt/spec.lpf_tau)
        self.latency = spec.latency_steps
        self.state = np.zeros((batch, 2, 2))
        self.lpf = np.zeros((batch, 2))
        self.command = np.zeros((batch, 2))
        self.pending = []
        self.history = np.zeros((buffer, batch, 2))
        self.step = 0

    def issue(self, now, command):
        self.pending.append((now+self.latency, command.copy()))

    def advance(self, until):
        """Compute filtered nominal contributions for all steps < until."""
        while self.step < until:
            while self.pending and self.pending[0][0] <= self.step:
                self.command = self.pending.pop(0)[1]
            angle = np.einsum('bij,bj->bi', self.gain, self.state[:, :, 0])
            self.lpf = angle if self.step == 0 else self.lpf+self.lpf_gain*(angle-self.lpf)
            self.history[self.step % len(self.history)] = self.lpf
            self.state = np.einsum('aij,baj->bai', self.ad, self.state)+self.command[:, :, None]*self.bd[None]
            self.step += 1

    def contribution(self, step):
        if not self.step-len(self.history) <= step < self.step:
            raise ValueError('Requested nominal contribution outside the buffer')
        return self.history[step % len(self.history)]


def lanczos(x, a):
    x = np.asarray(x, dtype=float)
    out = np.sinc(x)*np.sinc(x/a)
    return np.where(abs(x) < a, out, 0.0)


class PredictiveController:
    """command='lead': invert the lever at t + latency + T_u/2 + FSM group delay (approximation).
    command='pinn': solve (P1) with the Frozen-PINN command layer under the FSM dynamics."""
    def __init__(self, model, name, online=None, command='lead', pinn=None):
        if command not in ('lead', 'pinn') or (command == 'pinn' and pinn is None):
            raise ValueError('command must be lead, or pinn with a command-layer config')
        self.model, self.name, self.online_cfg = model, name, online
        self.command_mode, self.pinn_cfg = command, pinn

    def reset(self, context):
        spec = context.spec
        self.spec = spec
        self.f = spec.psd_lever
        self.inverse_gain = np.linalg.inv(context.gain_model)
        self.target = context.target
        b = context.batch
        self.plant = NominalPlant(spec, b, context.gain_model)
        self.frames = []
        self.captures = []
        self.features = {}
        self.sequence = None
        self.command = np.zeros((b, 2))
        cfg = self.online_cfg
        self.readout = OnlineReadout(self.model, b, cfg['window'] if cfg else 1, cfg['every'] if cfg else 1,
                                     cfg['prior'] if cfg else None)
        # Delay from tick to the mean time the new command acts on the beam, per axis, in steps;
        # plus the PSD filter delay, because d is reconstructed from filtered readings.
        self.lead = (spec.latency_steps+spec.tick_steps/2
                     + (spec.fsm_group_delay+spec.lpf_tau)/spec.dt)
        self.a = spec.controllers.lanczos_a
        self.layer = None
        if self.command_mode == 'pinn':
            from pat.command_layer import PINNCommandLayer
            if np.ptp(context.gain_model, axis=0).max() > 0:
                raise ValueError('Command layer is built once for one nominal C_F')
            self.layer = PINNCommandLayer(spec, context.gain_model[0], self.pinn_cfg)
            # Collocation times measured from the application instant, in steps, plus the PSD filter delay.
            self.layer_offsets = (self.layer.tau+spec.lpf_tau)/spec.dt

    def on_frame(self, reading, capture_step, now):
        self.plant.advance(now+1)
        d = reading/self.f-self.plant.contribution(capture_step)
        self.frames.append(d)
        self.captures.append(capture_step)
        n, h = self.model.history, self.model.horizon
        k = len(self.frames)-1
        if k+1 < n:
            # Before the history fills, predict by holding the latest sample (zero-order): the
            # command is still the one-shot inverse of the lever, from the first frame on.
            past = np.stack(self.frames[-3:], axis=1) if k >= 2 else np.repeat(d[:, None], 3, axis=1)
            self.sequence = (np.concatenate([past, np.repeat(d[:, None], h, axis=1)], axis=1), capture_step)
            return
        window = np.stack(self.frames[-n:], axis=1)
        x = self.model.features(self.model.inputs(window))
        self.features[k] = x
        future = d[:, None, :]+self.model.increments(self.readout.predict(x))
        past = np.stack(self.frames[-3:], axis=1) if k >= 2 else np.repeat(d[:, None], 3, axis=1)
        self.sequence = (np.concatenate([past, future], axis=1), capture_step)
        done = k-h
        if done in self.features:
            target = self.model.targets(self.frames[done], np.stack(self.frames[done+1:done+h+1], axis=1))
            self.readout.add(self.features.pop(done), target)
        if len(self.frames) > n+h+3:
            self.frames.pop(0)
            self.captures.pop(0)
            self.features = {key-1: value for key, value in self.features.items()}

    def _interpolate(self, seq, capture, steps):
        """Disturbance at the given absolute (fractional) steps; seq index 2 is the latest frame."""
        position = 2+(np.asarray(steps, dtype=float)-capture)/self.spec.frame_steps
        grid = np.arange(seq.shape[1])
        weights = lanczos(position[..., None]-grid, self.a)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights

    def tick(self, now):
        self.plant.advance(now+1)
        if self.sequence is not None:
            seq, capture = self.sequence
            if self.layer is None:
                weights = self._interpolate(seq, capture, now+self.lead)          # (2 axes, J)
                d = np.einsum('bjc,cj->bc', seq, weights)
                desired = self.target/self.f-d
                command = np.einsum('bij,bj->bi', self.inverse_gain, desired)
            else:
                spec = self.spec
                u_prev = self.plant.command
                state = self.plant.state.copy()
                # Nominal state when the new command reaches the mirror (previous command until then).
                for _ in range(spec.latency_steps-1):
                    state = np.einsum('aij,baj->bai', self.plant.ad, state)+u_prev[:, :, None]*self.plant.bd[None]
                weights = self._interpolate(seq, capture, now+spec.latency_steps+self.layer_offsets)  # (n, J)
                d_hat = np.einsum('bjc,nj->bnc', seq, weights)*self.f
                command = self.layer.command(state[:, :, 0], state[:, :, 1], u_prev, self.target, d_hat)
            self.command = np.clip(command, -self.spec.fsm.software_range_rad, self.spec.fsm.software_range_rad)
        self.plant.issue(now, self.command)
        return self.command


class OracleController:
    """Upper bound: true disturbance, offset and C_F at the effective time; jitter unknown."""
    name = 'oracle'

    def __init__(self, episodes, batch_index=None):
        self.episodes = episodes
        self.idx = np.arange(episodes.count) if batch_index is None else np.asarray(batch_index)

    def reset(self, context):
        spec = context.spec
        self.spec, self.f = spec, spec.psd_lever
        self.inverse_gain = np.linalg.inv(self.episodes.gain_true[self.idx])
        self.target = context.target
        self.lead = spec.latency_steps+spec.tick_steps/2+spec.fsm_group_delay/spec.dt

    def on_frame(self, reading, capture_step, now):
        pass

    def tick(self, now):
        alpha = self.episodes.alpha[self.idx]
        steps = np.clip(np.rint(now+self.lead).astype(int), 0, alpha.shape[1]-1)
        d = np.stack([alpha[:, steps[c], c] for c in range(2)], axis=1)
        desired = self.target/self.f-d
        return np.einsum('bij,bj->bi', self.inverse_gain, desired)-self.episodes.fsm_offset[self.idx]
