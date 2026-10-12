"""Recheck recorded physical data offline; never connect to any instrument."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from psd_position_pinn import PSDPositionController, PSDPositionPINN, train_psd_position_pinn

ROOT = Path(__file__).resolve().parent
FIXED = ROOT / "evidence" / "fixed_source_20261012"
DATASET = FIXED / "PSD_FSM_PINN_recalibration_20261012.json"
MODEL = FIXED / "PSD_FSM_PINN_recalibration_20261012_PINN_model.json"
TRIAL = FIXED / "PSD_PINN_20261012_013604.json"


def main():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    saved = PSDPositionPINN.load(MODEL)
    assert saved.record["simulation"] is False
    assert saved.record["reference_commit"] == "1192e7e92b885bfcb72e1c712948c8624cd3b3b5"
    trained = train_psd_position_pinn(dataset)
    rows = dataset["center_calibration_samples"]
    q = np.array([r["FSM_feedback_xy"] for r in rows])
    p = np.array([r["PSD_mm"] for r in rows])
    indices = saved.record["withheld_sample_indices"]
    predictions = np.array([saved.position_and_jacobian(v)[0] for v in q])
    retrained = np.array([trained.position_and_jacobian(v)[0] for v in q])
    withheld_rms = float(np.sqrt(np.mean((predictions[indices] - p[indices]) ** 2)))
    delta = float(np.max(np.abs(predictions - retrained)))
    assert abs(withheld_rms - saved.record["withheld_PSD_rms_mm"]) < 1e-8
    assert delta < 1e-7, "The pinned upstream basis no longer reproduces the recorded model"

    # Check the stored physical model's analytic derivative independently.
    derivative_errors = []
    for offset in ([0, 0], [.0003, -.0002], [-.0002, .0003]):
        probe = saved.center + offset
        assert saved.contains(probe)
        _, jacobian = saved.position_and_jacobian(probe)
        step = 1e-7
        finite_difference = np.column_stack([
            (saved.position_and_jacobian(probe + np.eye(2)[i] * step)[0]
             - saved.position_and_jacobian(probe - np.eye(2)[i] * step)[0]) / (2 * step)
            for i in range(2)
        ])
        derivative_errors.append(float(np.max(np.abs(jacobian - finite_difference))))
    assert max(derivative_errors) < 1e-4

    trial = json.loads(TRIAL.read_text(encoding="utf-8"))
    samples = trial["samples"]
    assert trial["sample_count"] == len(samples) == 889
    assert trial["reacquisition"]["state"] == "real_PD_PSD_reacquired"
    assert trial["reacquisition"]["PINN_used_during_search"] is False
    assert trial["source_stage_position_used"] is False
    controller = PSDPositionController(saved)
    command_errors = []
    # Replay both capture corrections and representative center-hold samples.
    replay_indices = sorted(set(range(12)) | set(np.linspace(12, len(samples) - 1, 16, dtype=int)))
    for index in replay_indices:
        row = samples[index]
        controller.reference = np.asarray(row["reference_PSD_mm"])
        command, _ = controller.next_command(row["PSD_mm"], row["FSM_feedback_xy"],
                                              row["previous_target_xy"])
        if row["PSD_error_norm_mm"] <= .015:
            command = np.asarray(row["previous_target_xy"])
        command_errors.append(float(np.max(np.abs(command - row["next_xy"]))))
    assert max(command_errors) <= 1.0001e-6

    confirmation = json.loads((FIXED / "PINN_reacquisition_verification_20261012_013604.json")
                              .read_text(encoding="utf-8"))
    assert confirmation["state"] == "passed"
    result = {
        "state": "passed",
        "validation_kind": "offline_retraining_and_replay_of_recorded_physical_measurements",
        "new_hardware_measurements_performed": False,
        "hardware_connections_or_commands": False,
        "upstream_commit": saved.record["reference_commit"],
        "calibration_sample_count": len(rows),
        "coefficient_fitting_sample_count": saved.record["training_sample_count"],
        "withheld_sample_count": len(indices),
        "withheld_component_RMS_mm": withheld_rms,
        "maximum_retrained_prediction_difference_mm": delta,
        "maximum_analytic_derivative_difference": max(derivative_errors),
        "recorded_trial_sample_count": len(samples),
        "recorded_trial_final_state": trial["state"],
        "recorded_trial_stop_reason": trial["error"],
        "recorded_reacquisition_elapsed_s": trial["reacquisition"]["elapsed_s"],
        "recorded_initial_center_hold_norm_RMS_mm": confirmation["recent_center_RMS_mm"],
        "recorded_initial_center_hold_sample_count": confirmation["recent_sample_count"],
        "replayed_command_sample_count": len(replay_indices),
        "maximum_recorded_command_replay_difference_xy": max(command_errors),
        "new_mounted_GBE_geometry_validated": False,
        "PD_PSD_center_coincidence_verified": False,
        "absolute_PSD_coordinate_accuracy_verified": False,
    }
    output = ROOT / "calibration" / "physical_dataset_replay_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
