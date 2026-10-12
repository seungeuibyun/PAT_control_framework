"""Find a repeatable local maximum of RIGOL CH1 mean voltage using the FSM.

Use the existing current mode and single-owner broker. PSD is recorded as a
diagnostic, not used as the objective. A browser Stop cancels the search lease
so later commands cannot restart the mirror. No laser or stage commands.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import uuid

from shared_acquisition import Broker


class SearchCancelled(RuntimeError):
    pass


class PDSearch:
    def __init__(self, args):
        self.args, self.broker = args, Broker(args.broker_state)
        self.search_id = uuid.uuid4().hex
        self.client = "pd-maximum-" + self.search_id[:12]
        self.sequence, self.active = 0, False
        self.scale = None
        self.original = None
        self.origin_xy = None
        self.result = {"started": dt.datetime.now().astimezone().isoformat(),
                       "state": "in_progress", "objective": "RIGOL_CH1_mean_voltage_V",
                       "search_id": self.search_id, "points": [],
                       "laser_commands_issued": False, "stage_commands_issued": False,
                       "electrical_dark_calibrated": False, "PD_optimal_PSD_reference_calibrated": False,
                       "max_current_delta_per_axis_mA": args.radius_mA,
                       "max_feedback_displacement": .30}
        self.output = Path(__file__).resolve().parent / (
            "PD_maximum_search_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")

    def save(self):
        self.result["updated"] = dt.datetime.now().astimezone().isoformat()
        self.output.write_text(json.dumps(self.result, indent=2, allow_nan=False), encoding="utf-8")

    def state(self, expected=None):
        s = self.broker.state()
        f = s["fsm"]
        if self.active and not f.get("automatic_search_active"):
            raise SearchCancelled("Search stopped by the viewer, watchdog or device fault")
        if not f.get("connected") or f.get("age_s", 9) >= 2 or f.get("fault"):
            raise RuntimeError("Fresh FSM feedback is unavailable: " + str(f.get("fault")))
        if f.get("operation_mode") != 0 or f.get("device_status", -1) & ~(1 << 4):
            raise RuntimeError("FSM operation mode or device status changed")
        if self.origin_xy and math.dist(f["feedback_xy"], self.origin_xy) > .30:
            raise RuntimeError("Local mirror displacement exceeded the search bound")
        if expected and max(abs(a-b) for a,b in zip(f["target_current_mA"], expected)) > .01:
            raise SearchCancelled("FSM command changed outside this search")
        return s

    def command(self, target):
        if max(abs(a-b) for a,b in zip(target, self.original)) > self.args.radius_mA + 1e-6:
            raise ValueError("Requested point exceeds the search current window")
        self.state()
        self.sequence += 1
        self.broker.post("/internal/fsm", {"action": "set", "x_mA": target[0],
            "y_mA": target[1], "client_id": self.client, "sequence": self.sequence,
            "search_id": self.search_id})
        deadline, previous, stable = time.monotonic()+3, None, 0
        while time.monotonic() < deadline:
            s = self.state(target)
            f = s["fsm"]
            reached = max(abs(a-b) for a,b in zip(f["applied_current_mA"], target)) < .05
            still = previous is not None and math.dist(previous, f["feedback_xy"]) < .0003
            stable = stable+1 if reached and still else 0
            if stable >= 2:
                time.sleep(.15)
                return
            previous = f["feedback_xy"]
            time.sleep(.10)
        raise RuntimeError("FSM did not settle at the requested current")

    def change_scale(self):
        scales = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10]
        chosen = next((v for v in scales if v > self.scale * 1.001), None)
        if chosen is None:
            raise RuntimeError("CH1 is outside the available voltage range")
        self.broker.post("/internal/scope", {"command": f":CHANnel1:SCALe {chosen}"})
        self.result.setdefault("scope_range_changes", []).append(
            {"time": dt.datetime.now().astimezone().isoformat(), "V_per_div": chosen})
        self.scale = chosen
        time.sleep(.3)

    def voltage(self, expected):
        while True:
            self.state(expected)
            try:
                value = self.broker.post("/internal/pd", {})["voltage_V"]
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")
                if "mean-voltage measurement is unavailable" not in message:
                    raise RuntimeError(message) from exc
                self.change_scale()
                continue
            if not math.isfinite(value):
                raise RuntimeError("Non-finite CH1 voltage")
            if abs(value) > 3 * self.scale:
                self.change_scale()
                continue
            if value >= 8:
                raise RuntimeError("PD output is near saturation; a maximum cannot be resolved")
            return value

    def point(self, target, label, count=4):
        self.command(target)
        readings = []
        for _ in range(count):
            readings.append(self.voltage(target))
            time.sleep(.08)
        s = self.state(target)
        p = {"index": len(self.result["points"]), "label": label,
             "time": dt.datetime.now().astimezone().isoformat(),
             "current_mA": list(target), "feedback_xy": s["fsm"]["feedback_xy"],
             "PD_mean_V": statistics.mean(readings), "PD_std_V": statistics.stdev(readings),
             "PD_samples_V": readings, "scope_scale_V_per_div": self.scale,
             "PSD": {k: s["psd"].get(k) for k in ["x_mm", "y_mm", "relative_percent", "position_available", "signal_state"]}}
        self.result["points"].append(p)
        self.save()
        best = max(self.result["points"], key=lambda row: row["PD_mean_V"])
        if p["index"] % 4 == 0 or label.startswith("verify"):
            print(json.dumps({"point": p["index"], "current_mA": target,
                              "PD_mV": round(p["PD_mean_V"]*1000,4),
                              "best_mV": round(best["PD_mean_V"]*1000,4)}), flush=True)
        return p

    def run(self):
        held, cancelled = False, False
        try:
            s = self.state()
            if not s["pd"].get("identity", "").startswith("RIGOL TECHNOLOGIES,MSO1104Z,"):
                raise RuntimeError("Verified RIGOL oscilloscope is unavailable")
            if self.broker.post("/internal/scope", {"command": ":TRIGger:STATus?"})["reply"] == "STOP":
                raise RuntimeError("Oscilloscope acquisition is stopped; no live objective available")
            self.scale = float(self.broker.post("/internal/scope", {"command": ":CHANnel1:SCALe?"})["reply"])
            self.result["original_scope_scale_V_per_div"] = self.scale
            self.original = s["fsm"]["target_current_mA"]
            self.origin_xy = s["fsm"]["feedback_xy"]
            self.result.update(original_current_mA=self.original, original_feedback_xy=self.origin_xy)
            self.broker.post("/internal/fsm-search", {"action": "begin", "search_id": self.search_id,
                                                     "client_id": self.client})
            self.active = True
            baseline = self.point(self.original, "initial_baseline", count=8)
            self.result["initial_PD_mean_V"] = baseline["PD_mean_V"]
            seen = {(0.,0.)}
            for radius, step in ((5.,5.),(10.,5.),(self.args.radius_mA, self.args.radius_mA/2)):
                values = [-radius+i*step for i in range(round(2*radius/step)+1)]
                for j, y in enumerate(values):
                    for x in values if j%2 == 0 else reversed(values):
                        if (x,y) in seen:
                            continue
                        seen.add((x,y))
                        self.point([self.original[0]+x,self.original[1]+y], "coarse_grid")
            best = max(self.result["points"], key=lambda row: row["PD_mean_V"])
            threshold = max(.001, 6*baseline["PD_std_V"])
            self.result["minimum_improvement_V"] = threshold
            if best["PD_mean_V"]-baseline["PD_mean_V"] >= threshold:
                for step in (2., 1., .5):
                    center = best["current_mA"]
                    for dx,dy in ((0,0),(-step,-step),(0,-step),(step,-step),
                                  (step,0),(step,step),(0,step),(-step,step),(-step,0)):
                        target = [center[0]+dx, center[1]+dy]
                        if max(abs(a-b) for a,b in zip(target,self.original)) <= self.args.radius_mA:
                            p = self.point(target, "refine")
                            if p["PD_mean_V"] > best["PD_mean_V"]:
                                best = p
                reference1 = self.point(self.original, "verify_reference_1", count=8)
                peak1 = self.point(best["current_mA"], "verify_peak_1", count=8)
                reference2 = self.point(self.original, "verify_reference_2", count=8)
                peak2 = self.point(best["current_mA"], "verify_peak_2", count=10)
                noise = max(reference1["PD_std_V"], reference2["PD_std_V"], peak1["PD_std_V"], peak2["PD_std_V"])
                reference = max(reference1["PD_mean_V"], reference2["PD_mean_V"])
                minimum = max(.001, 6*noise)
                verified = min(peak1["PD_mean_V"],peak2["PD_mean_V"])-reference >= minimum
                self.result["verification"] = {"passed": verified, "reference_mean_V": reference,
                    "repeat_peak_mean_V": [peak1["PD_mean_V"],peak2["PD_mean_V"]],
                    "minimum_improvement_V": minimum, "repeat_peak_feedback_xy": [peak1["feedback_xy"],peak2["feedback_xy"]]}
                if verified:
                    held = True
                    self.result.update(state="verified_local_PD_maximum_held", held_point=peak2,
                                       best_observed_point=best)
                else:
                    self.result.update(state="peak_not_repeatable", best_observed_point=best)
            else:
                self.result.update(state="no_distinguishable_PD_peak_in_search_window", best_observed_point=best)
        except SearchCancelled as exc:
            cancelled = True
            self.result.update(state="cancelled", error=str(exc))
        except Exception as exc:
            self.result.update(state="aborted", error=str(exc))
        finally:
            if self.active:
                try:
                    if not held and not cancelled:
                        self.command(self.original)
                    end = self.broker.post("/internal/fsm-search", {"action": "end",
                        "search_id": self.search_id, "client_id": self.client})
                    self.result["handoff"] = end
                    if held and (not end.get("accepted") or end.get("monitor_ip") != self.args.viewer_ip):
                        self.result["state"] = "verified_peak_found_but_not_held"
                    self.active = False
                except Exception as exc:
                    self.result["cleanup_error"] = str(exc)
                    # The broker watchdog zeros current if the search cannot release.
            try:
                final = self.broker.state()
                self.result["final_FSM"] = final["fsm"]
                self.result["final_PD"] = final["pd"]
            except Exception as exc:
                self.result["final_status_error"] = str(exc)
            self.result["finished"] = dt.datetime.now().astimezone().isoformat()
            self.save()
            print(json.dumps({"saved": str(self.output), "state": self.result["state"],
                "before_mV": self.result.get("initial_PD_mean_V",0)*1000,
                "best_mV": self.result.get("best_observed_point",{}).get("PD_mean_V",0)*1000,
                "held_mV": self.result.get("held_point",{}).get("PD_mean_V",0)*1000,
                "error": self.result.get("error"), "cleanup_error": self.result.get("cleanup_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker-state", default=str(Path(__file__).with_name("live_gui_connection.json")))
    parser.add_argument("--viewer-ip", default="192.168.0.36")
    parser.add_argument("--radius-mA", type=float, default=20.)
    args = parser.parse_args()
    if not math.isfinite(args.radius_mA) or not 10 <= args.radius_mA <= 50:
        parser.error("Search radius must be between 10 and 50 mA per axis")
    PDSearch(args).run()
