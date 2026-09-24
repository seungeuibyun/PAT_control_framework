"""Independent SSFM entrance field, cached per interval; receiver-only autograd."""
from __future__ import annotations
import time
import numpy as np
import torch
from solver.optimization import projected_gradient_ascent, sensing_initial_command


def _resolve_device(device):
    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps or auto")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable in this Python process. Check the PyTorch build and GPU access.")
    return resolved


class TorchReceiver:
    """Differentiable receiver on CPU/CUDA/MPS.

    MPS represents every complex array as two float32 tensors. The Fourier
    matrices, FSM phase and intensity are evaluated with real arithmetic;
    autograd never encounters a complex MPS tensor. ``real_pair`` can also be
    selected on CPU to check this arithmetic against the native complex path.
    """
    def __init__(self, system, device="cpu", dtype="float64", *, complex_backend="auto"):
        self.system = system
        self.device = _resolve_device(device)
        if dtype not in {"float64", "float32"}:
            raise ValueError("dtype must be float64 or float32")
        self.requested_dtype = dtype
        self.effective_dtype = "float32" if self.device.type == "mps" else dtype
        self.real_dtype = torch.float64 if self.effective_dtype == "float64" else torch.float32
        self.complex_dtype = torch.complex128 if self.effective_dtype == "float64" else torch.complex64
        if complex_backend == "auto":
            complex_backend = "real_pair" if self.device.type == "mps" else "native"
        if complex_backend not in {"native", "real_pair"}:
            raise ValueError("complex_backend must be auto, native or real_pair")
        if self.device.type == "mps" and complex_backend != "real_pair":
            raise ValueError("Use real_pair complex representation on MPS")
        self.complex_backend = complex_backend
        self.X, self.Y = self.tensor(system.RX), self.tensor(system.RY)
        self.CF = self.tensor(system.CF)
        self.ey, self.ex = (self.tensor(m, True) for m in system.detector_operators)
        self.mask = self.tensor(system.detector_mask)
        o = system.cfg.optical
        self.scale = np.sqrt(o.splitter_transmission) * system.rdx**2 / (o.wavelength * o.focal_length)
        self.reduced = None

    def tensor(self, value, complex_value=False):
        if complex_value and self.complex_backend == "real_pair":
            array = np.asarray(value)
            return self.tensor(array.real), self.tensor(array.imag)
        # Cast on the host before transfer: no float64 or complex tensor is
        # ever allocated on MPS, including strided NumPy .real/.imag views.
        numpy_dtype = (np.complex128 if self.effective_dtype == "float64" else np.complex64) if complex_value else (
            np.float64 if self.effective_dtype == "float64" else np.float32)
        array = np.ascontiguousarray(value, dtype=numpy_dtype)
        return torch.as_tensor(array, dtype=self.complex_dtype if complex_value else self.real_dtype, device=self.device)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def set_field(self, reduced):
        self.reduced = self.tensor(reduced, True)
        self._sync()

    def propagate(self, theta):
        """Return the detector field (real/imaginary tensor pair on MPS)."""
        angle = self.CF @ theta
        phase = self.system.cfg.optical.k0 * (angle[0] * self.X + angle[1] * self.Y)
        if self.complex_backend == "real_pair":
            ur, ui = self.reduced
            cosine, sine = torch.cos(phase), torch.sin(phase)
            fr, fi = ur * cosine - ui * sine, ur * sine + ui * cosine
            yr, yi = self.ey
            xr, xi = self.ex
            # (Ey @ F) @ Ex.T, expanded into real matrix products.
            tr, ti = yr @ fr - yi @ fi, yr @ fi + yi @ fr
            return (self.scale * (tr @ xr.T - ti @ xi.T),
                    self.scale * (tr @ xi.T + ti @ xr.T))
        F = self.reduced * torch.exp(1j * phase)
        return self.scale * (self.ey @ F @ self.ex.T)

    def _power(self, theta):
        field = self.propagate(theta)
        if self.complex_backend == "real_pair":
            real, imag = field
            intensity = real.square() + imag.square()
        else:
            intensity = torch.abs(field)**2
        return torch.sum(self.mask * intensity) * self.system.cfg.optical.detector_sampling**2

    def objective_and_gradient(self, theta):
        command = self.tensor(theta).detach().requires_grad_(True)
        value = self._power(command)
        gradient, = torch.autograd.grad(value, command)
        self._sync()
        return value.item(), gradient.detach().cpu().numpy().astype(float)

    def objective_only(self, theta):
        with torch.no_grad():
            value = self._power(self.tensor(theta))
        self._sync()
        return value.item()


class DifferentiableSSFMSolver:
    name = "Diff-SSFM oracle"

    def __init__(self, system, cfg, *, device=None, dtype=None):
        self.system, self.cfg = system, cfg
        self.engine = TorchReceiver(system, device or cfg.baselines.ssfm_device, dtype or cfg.baselines.ssfm_dtype)
        self.device = self.engine.device
        self.requested_dtype = self.engine.requested_dtype
        self.effective_dtype = self.engine.effective_dtype
        self._interval = None
        self.prepare_time = 0.0

    def _sync(self):
        self.engine._sync()

    def prepare_interval(self, interval):
        if self._interval == interval:
            return
        start = time.perf_counter()
        # Independent numerical propagation, charged once to online latency.
        self._reduced = self.system.reduce_field(self.system.atmospheric_field(interval, cache=False))
        self.engine.set_field(self._reduced)
        self._interval = interval
        self.prepare_time = time.perf_counter() - start

    def objective_and_gradient(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.engine.objective_and_gradient(theta)

    def objective_only(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.engine.objective_only(theta)

    def solve(self, interval, target, theta_prev, history, measurement=None, measurement_valid=True):
        self._sync()
        start = time.perf_counter()
        self.prepare_interval(interval)
        initial = sensing_initial_command(self.system, theta_prev, measurement, measurement_valid)
        query = time.perf_counter()
        theta, value, diag = projected_gradient_ascent(theta_prev,
            lambda command: self.objective_and_gradient(command, interval),
            lambda command: self.objective_only(command, interval), self.cfg.tracking, initial)
        self._sync()
        diag.update(device=str(self.device), requested_dtype=self.requested_dtype,
                    effective_dtype=self.effective_dtype, receiver_complex_backend=self.engine.complex_backend,
                    atmosphere_prepare_time_sec=self.prepare_time,
                    atmosphere_integrations_this_interval=1, atmosphere_depends_on_fsm=False,
                    atmosphere_backend="numpy_cpu_ssfm", gradient_backend="torch_receiver_autograd",
                    initial_command=initial, query_runtime_sec=time.perf_counter() - query)
        return dict(theta=theta, predicted_metric=value, runtime_sec=time.perf_counter() - start, diagnostics=diag)
