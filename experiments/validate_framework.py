#!/usr/bin/env python3
"""Physics/regression checks for the receiver-side PDF implementation."""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config.settings import make_preset
from system_model.optical_system import OpticalPATSystem
from solver.optimization import project_pat_command, quantize_pat_command, projected_gradient_ascent
from solver.frozen_pinn import FrozenPINNSolver
from solver.differentiable_ssfm import TorchReceiver, _resolve_device


def assert_close(name, actual, expected, atol=0, rtol=1e-5):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    print(f"PASS {name}")


def make_cfg():
    c = make_preset("demo")
    c.optical.grid_size = 256
    c.turbulence.strength_scale = 0
    c.tracking.disturbance = "zero"
    c.tracking.num_intervals = 2
    return c


def test_constraints():
    c = make_cfg()
    t = c.tracking
    prev = np.array([999e-6, -998e-6])
    got = project_pat_command([2e-3, -2e-3], prev, t)
    assert_close("axis-wise range", got, [1e-3, -1e-3], atol=1e-15)
    got = project_pat_command([2e-3, 2e-3], [0, 0], t)
    assert_close("axis-wise slew (not L2 ball)", got, [100e-6, 100e-6], atol=1e-15)
    t.theta_slew_max_override = 0.4e-6
    assert_close("quantization with sub-grid slew", quantize_pat_command([1e-3, 1e-3], prev, t), prev, atol=1e-15)
    t.theta_slew_max_override = 100.3e-6
    assert_close("non-grid box endpoint", quantize_pat_command([1e-3, 1e-3], [0, 0], t), [100e-6]*2, atol=1e-15)
    optimum = np.array([34.4e-6, -58.2e-6])
    fun = lambda x: -np.sum((x-optimum)**2)
    cmd, value, diag = projected_gradient_ascent(np.zeros(2), lambda x:(fun(x),-2*(x-optimum)), fun, t)
    assert_close("post-quantization prediction", value, fun(cmd), atol=1e-18)
    assert np.all(np.diff(diag["continuous_power_trace_w"]) >= -1e-18)


def test_physics():
    c = make_cfg()
    s = OpticalPATSystem(c)
    u = s.atmospheric_field(0)
    assert_close("SSFM extinction/power", (abs(u)**2).sum()*s.dx**2,
                 c.optical.transmit_power*10**(-c.optical.path_loss_db/10), rtol=1e-8)
    o = c.optical
    zR = np.pi*o.beam_waist**2/o.wavelength
    width = o.beam_waist*np.sqrt(1+(o.propagation_distance/zR)**2)
    expected = np.sqrt(2*o.transmit_power/(np.pi*width**2))*np.exp(-(s.X**2+s.Y**2)/width**2)*10**(-o.path_loss_db/20)
    assert_close("Gaussian diffraction amplitude", abs(u), expected, atol=1e-8, rtol=1e-4)
    reduced = s.reduce_field(u)
    analytic_clipped = o.transmit_power*10**(-o.path_loss_db/10)*(1-np.exp(-2*(o.entrance_diameter/2)**2/width**2))
    assert_close("aperture + reducer area normalization", (abs(reduced)**2).sum()*s.rdx**2,
                 o.reducer_transmission*analytic_clipped, rtol=0.002)
    p0 = s.sensing_vector(reduced, [0,0])
    p1 = s.sensing_vector(reduced, [30e-6, -20e-6])
    assert_close("FSM changes phase, preserves mirror intensity", abs(s.fsm_field(reduced,[30e-6,-20e-6])),abs(reduced),atol=1e-14)
    assert_close("focal-plane reflection gain", p1[:2]-p0[:2], s.psd_jacobian @ [30e-6,-20e-6], atol=0.15e-6)
    assert p0[2] <= o.splitter_reflection*(abs(reduced)**2).sum()*s.rdx**2*(1+1e-8)
    theta = np.array([45e-6,-35e-6]); value, gradient = s.power_and_gradient(reduced, theta)
    h=1e-8
    fd=np.array([(s.power(s.detector_field(reduced,theta+h*a))-s.power(s.detector_field(reduced,theta-h*a)))/(2*h) for a in np.eye(2)])
    assert_close("Eq. (42) receiver analytic gradient vs finite difference",gradient,fd,rtol=1e-6,atol=1e-9)
    engine=TorchReceiver(s);engine.set_field(reduced)
    qt,gt=engine.objective_and_gradient(theta)
    assert_close("Torch/NumPy receiver power",qt,value,rtol=1e-10)
    assert_close("Torch autograd/analytic receiver gradient",gt,gradient,rtol=1e-9,atol=1e-10)
    c.optical.psd_min_power=1
    measured, valid=s.measure_psd(reduced,[0,0],0)
    assert not valid
    print("PASS invalid PSD reading is flagged")


def test_frozen():
    c=make_cfg()
    # A smaller, still resolved setting makes this a fast integration check.
    c.frozen_pinn.hidden_width=256
    c.frozen_pinn.collocation_side=48
    c.frozen_pinn.spectral_side=24
    s=OpticalPATSystem(c);f=FrozenPINNSolver(s,c)
    f.prepare_interval(0)
    count=f.atmosphere_integrations
    f.objective_and_gradient([0,0],0)
    f.objective_and_gradient([1e-5,-1e-5],0)
    assert f.atmosphere_integrations==count==1
    print("PASS atmospheric ODE reused across FSM commands")
    assert_close("Frozen coefficient norm follows extinction",f.interval_diagnostics["coefficient_power_ratio"],
                 10**(-c.optical.path_loss_db/10),rtol=2e-4)
    boundary=f.basis.reconstruct(f._coefficients,np.array([-1.,1.]),np.linspace(-1,1,31))
    assert_close("continuous zero-field boundary",boundary,0,atol=1e-12)
    before=s.tx_field(0).copy();s.fsm_field(f._reduced,[1e-3,-1e-3])
    assert_close("transmitter independent of FSM",s.tx_field(0),before,rtol=0,atol=0)


def test_sampling_and_randomness():
    c=make_cfg();c.turbulence.strength_scale=1;c.tracking.disturbance="iid"
    s=OpticalPATSystem(c)
    a=s.turbulence.eval(s.X,s.Y,19980,0)
    b=s.turbulence.eval(s.X,s.Y,19980,1)
    assert not np.allclose(a,b,rtol=1e-8,atol=1e-18)
    assert_close("channel replay independent of solver order",s.turbulence.eval(s.X,s.Y,19980,0),a,rtol=0,atol=0)
    c2=deepcopy(c);c2.tracking.num_intervals=100
    s2=OpticalPATSystem(c2)
    assert_close("interval count does not alter Tx sequence",s.tx_angle(3),s2.tx_angle(3),rtol=0,atol=0)
    c2.optical.receiver_grid_size=64
    try:
        c2.validate()
    except ValueError:
        print("PASS unresolved PSD grid rejected")
    else:
        raise AssertionError("Aliased PSD grid accepted")


def test_receiver_device(device="cpu"):
    """Exercise the exact MPS arithmetic even when running on a CPU host."""
    c = make_cfg()
    c.optical.fsm_calibration = ((2.0, 0.15), (-0.1, 1.9))
    s = OpticalPATSystem(c)
    reduced = s.reduce_field(s.atmospheric_field(0))
    native = TorchReceiver(s, device="cpu", dtype="float64")
    paired = TorchReceiver(s, device="cpu", dtype="float64", complex_backend="real_pair")
    selected = TorchReceiver(s, device=device, dtype="float64")
    paired32 = TorchReceiver(s, device="cpu", dtype="float32", complex_backend="real_pair")
    for engine in (native, paired, selected, paired32):
        engine.set_field(reduced)
    for theta in ([45e-6, -35e-6], [-125e-6, 90e-6], [200e-6, -150e-6]):
        value, gradient = s.power_and_gradient(reduced, theta)
        for label, engine in (("native CPU", native), ("real-pair CPU", paired),
                              ("real-pair CPU float32", paired32), (f"selected {selected.device}", selected)):
            power, grad = engine.objective_and_gradient(theta)
            low_precision = engine.effective_dtype == "float32"
            tolerance = 3e-4 if low_precision else 1e-9
            assert_close(f"{label} receiver power", power, value, rtol=tolerance)
            assert_close(f"{label} receiver gradient", grad, gradient, rtol=tolerance, atol=1e-7 if low_precision else 1e-10)
            assert_close(f"{label} objective_only", engine.objective_only(theta), power, rtol=tolerance)
    if selected.device.type == "mps":
        assert selected.requested_dtype == "float64"
        assert selected.effective_dtype == "float32"
        assert selected.complex_backend == "real_pair"
        tensors = (selected.X, selected.Y, selected.CF, selected.mask,
                   *selected.ex, *selected.ey, *selected.reduced)
        assert all(t.device.type == "mps" and t.dtype == torch.float32 for t in tensors)
        assert all(t.device.type == "mps" and t.dtype == torch.float32
                   for t in selected.propagate(selected.tensor([0., 0.])))
        print("PASS actual MPS tensors, float64-to-float32 selection and real-pair autograd")


def test_device_selection_and_cli():
    # Check auto selection independently of the test host's hardware.
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
        assert _resolve_device("auto").type == "mps"
        assert _resolve_device("mps").type == "mps"
    with patch("torch.cuda.is_available", return_value=True), patch("torch.backends.mps.is_available", return_value=True):
        assert _resolve_device("auto").type == "cuda"
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
        assert _resolve_device("auto").type == "cpu"
        try:
            _resolve_device("mps")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Explicit MPS request silently fell back to CPU")
    from experiments.compare_pat import build_parser, make_config
    from experiments.runtime_scaling import build_parser as runtime_parser
    cfg = make_config(build_parser().parse_args(["--device", "mps"]))
    assert cfg.frozen_pinn.device == cfg.baselines.ssfm_device == "mps"
    assert cfg.frozen_pinn.backend == "auto"
    defaults = build_parser().parse_args([])
    assert defaults.preset == "paper" and defaults.gif
    assert cfg.optical.grid_size == 1024 and cfg.frozen_pinn.hidden_width == 1024
    assert not build_parser().parse_args(["--no-gif"]).gif
    assert runtime_parser().parse_args(["--device", "mps"]).device == "mps"
    print("PASS MPS CLI options, default Torch backend and automatic device selection")


def test_projected_evolution(device="cpu"):
    """Independent matrix-free RHS and full RK45 trajectory equivalence."""
    from solver.frozen_pinn_torch import TorchFrozenPINNSolver
    c = make_cfg()
    c.turbulence.strength_scale = 1
    c.tracking.disturbance = "iid"
    c.frozen_pinn.hidden_width = 128
    c.frozen_pinn.collocation_side = 48
    c.frozen_pinn.spectral_side = 24
    c.frozen_pinn.operator_backend = "projected"
    c.frozen_pinn.device = device
    s = OpticalPATSystem(c)
    ref = FrozenPINNSolver(s, c)
    fast = TorchFrozenPINNSolver(s, c)
    b = ref.basis
    rng = np.random.default_rng(82)
    state = rng.normal(size=b.R) + 1j*rng.normal(size=b.R)
    # Near-ground turbulence exercises the strongest potential.
    dn = s.turbulence.eval(b.X, b.Y, c.optical.propagation_distance-1, 1)
    generator = fast._layer_generator(dn)
    rhs = 1j*b._multiply(generator, state)-c.optical.attenuation/2*state
    reference_rhs = b.rhs(state, dn)
    tolerance = 5e-4 if fast.engine.effective_dtype == "float32" else 1e-9
    relative_rhs = np.linalg.norm(rhs-reference_rhs)/np.linalg.norm(reference_rhs)
    assert relative_rhs < tolerance, relative_rhs
    for interval in (0, 1):
        ref.prepare_interval(interval); fast.prepare_interval(interval)
        error = np.linalg.norm(fast._coefficients-ref._coefficients)/np.linalg.norm(ref._coefficients)
        assert error < tolerance, error
        for command in ([0., 0.], [35e-6, -25e-6]):
            assert_close("projected RK45 receiver power vs reference", fast.objective_only(command, interval),
                         ref.objective_only(command, interval), rtol=5*tolerance)
    assert fast.spatial_basis.device.type == fast.engine.device.type
    print(f"PASS {fast.engine.device} projected potential / matrix-free RHS and RK45 trajectories")


def test_device_controllers(device="cpu"):
    from simulation.runner import build_solvers
    c = make_cfg()
    c.frozen_pinn.hidden_width = 128
    c.frozen_pinn.collocation_side = 48
    c.frozen_pinn.spectral_side = 24
    c.frozen_pinn.device = c.baselines.ssfm_device = device
    c.tracking.max_opt_iterations = 2
    s = OpticalPATSystem(c)
    solvers = build_solvers(s, c, ["frozen_pinn", "ssfm_oracle"])
    for solver in solvers:
        previous = np.zeros(2)
        for interval in range(2):
            truth = s.reduce_field(s.atmospheric_field(interval))
            measurement, valid = s.measure_psd(truth, previous, interval)
            result = solver.solve(interval, s.reference_centroid, previous, [], measurement, valid)
            theta = result["theta"]
            assert np.all(np.isfinite(theta)) and np.isfinite(result["predicted_metric"])
            assert np.all(abs(theta-previous) <= c.tracking.theta_slew_max + 1e-12)
            assert_close("controller quantization", theta/c.tracking.theta_quantization,
                         np.round(theta/c.tracking.theta_quantization), atol=1e-8, rtol=0)
            diag = result["diagnostics"]
            assert diag["device"] == str(solver.engine.device)
            assert diag["requested_dtype"] == "float64"
            assert diag["effective_dtype"] == solver.engine.effective_dtype
            assert diag["receiver_complex_backend"] == solver.engine.complex_backend
            if solver.engine.device.type == "mps":
                assert diag["effective_dtype"] == "float32" and diag["receiver_complex_backend"] == "real_pair"
            # Both wrappers must reuse the prepared channel for new commands.
            old_field = solver.engine.reduced
            solver.objective_only(theta, interval)
            assert solver.engine.reduced is old_field
            previous = theta
        if isinstance(solver, FrozenPINNSolver):
            assert solver.atmosphere_integrations == 2
        print(f"PASS {solver.name} two-interval controller on {solver.engine.device}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu",
                        help="Also verify receiver arithmetic and both model-based controllers on this device")
    args = parser.parse_args()
    test_device_selection_and_cli()
    test_constraints();test_physics();test_frozen();test_sampling_and_randomness()
    test_receiver_device(args.device)
    test_projected_evolution(args.device)
    test_device_controllers(args.device)
    print("All receiver-PAT checks passed.")


if __name__ == "__main__":
    main()
