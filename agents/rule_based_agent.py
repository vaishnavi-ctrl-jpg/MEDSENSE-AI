"""
MedSense AI — Rule-Based Agent (Nidz's file, fixed)
====================================================

BUGS FIXED:
  1. When obs was a numpy array, bp_sys (obs[0]) was never extracted!
     The original code jumped straight to hr=obs[2]. bp_sys check was
     present in logic but the variable was undefined → NameError crash.

  2. Complaint set {7, 8, 9} — complaint 9 is "Minor Laceration" which
     should be DELAY not REFER. Fixed to {7, 8} matching ViVi's grader.

  3. act() signature used env=None — works fine, kept as is.
"""

import numpy as np
from medsense.constants import TREAT_NOW, DELAY, REFER, THRESHOLDS, OBS_IDX


class RuleBasedAgent:
    """
    Baseline clinical rules agent — the 'Human Benchmark'.
    Accepts both dict (API calls) and numpy array (RL env) observations.
    """

    def __init__(self):
        self.name = "Clinical Rules Baseline"

    def act(self, obs, env=None) -> int:
        """
        Returns triage action based on clinical threshold rules.
        Works with both dict observations (from API) and numpy arrays (from env).
        """
        if isinstance(obs, dict):
            # Dict format — from API / manual calls
            bp_sys    = obs.get("blood_pressure_systolic", 120)
            hr        = obs.get("heart_rate", 80)
            spo2      = obs.get("spo2", 98)
            temp      = obs.get("temperature", 37.0)
            resp_rate = obs.get("resp_rate", 16)
            complaint = int(obs.get("chief_complaint", 0))
            cardiac   = obs.get("has_cardiac_history", 0)
        else:
            # Numpy array — from MedSenseEnv.step()
            # BUG FIX: was missing bp_sys extraction entirely
            bp_sys    = obs[OBS_IDX["bp_systolic"]]
            hr        = obs[OBS_IDX["heart_rate"]]
            spo2      = obs[OBS_IDX["spo2"]]
            temp      = obs[OBS_IDX["temperature"]]
            resp_rate = obs[OBS_IDX["resp_rate"]]
            complaint = int(obs[OBS_IDX["complaint"]])
            cardiac   = obs[OBS_IDX["cardiac"]]

        # ── Clinical rules ────────────────────────────────────────────────────

        # Rule 1: Oxygen / blood pressure
        if spo2 < THRESHOLDS["spo2_min"]:
            return TREAT_NOW
        if bp_sys > THRESHOLDS["sys_bp_max"] or bp_sys < THRESHOLDS["sys_bp_min"]:
            return TREAT_NOW

        # Rule 2: Heart rate / temperature
        if hr > THRESHOLDS["hr_max"] or hr < THRESHOLDS["hr_min"]:
            return TREAT_NOW
        if temp > THRESHOLDS["temp_max"]:
            return TREAT_NOW

        # Rule 3: Respiratory rate
        if resp_rate > 25:
            return TREAT_NOW

        # Rule 4: Cardiac history + elevated vitals
        if cardiac == 1 and (hr > 110 or bp_sys > 155):
            return TREAT_NOW

        # Rule 5: Specialist complaints → refer
        # BUG FIX: removed complaint 9 (Minor Laceration → should be DELAY)
        if complaint in {7, 8}:
            return REFER

        # Rule 6: Borderline elevated — still needs treatment
        if bp_sys > 150 or hr > 105:
            return TREAT_NOW

        return DELAY


# ── Convenience function for grader compatibility ─────────────────────────────
def rule_based_policy(obs: np.ndarray, env) -> int:
    """Functional wrapper — grader expects fn(obs, env) not a class method."""
    return RuleBasedAgent().act(obs, env)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = RuleBasedAgent()

    tests = [
        ({"spo2": 85,  "blood_pressure_systolic": 120, "heart_rate": 80,  "temperature": 37.0, "resp_rate": 16, "chief_complaint": 1, "has_cardiac_history": 0}, TREAT_NOW, "Critical SpO2"),
        ({"spo2": 98,  "blood_pressure_systolic": 195, "heart_rate": 90,  "temperature": 37.0, "resp_rate": 16, "chief_complaint": 0, "has_cardiac_history": 1}, TREAT_NOW, "Hypertensive crisis"),
        ({"spo2": 97,  "blood_pressure_systolic": 115, "heart_rate": 72,  "temperature": 37.0, "resp_rate": 15, "chief_complaint": 9, "has_cardiac_history": 0}, DELAY,     "Stable patient"),
        ({"spo2": 97,  "blood_pressure_systolic": 115, "heart_rate": 72,  "temperature": 37.0, "resp_rate": 15, "chief_complaint": 7, "has_cardiac_history": 0}, REFER,     "Fracture → refer"),
    ]

    print("\nRule-Based Agent Test:")
    all_passed = True
    for obs, expected, label in tests:
        result = agent.act(obs)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} {label}: got {result}, expected {expected}")

    print(f"\n  {'All tests passed!' if all_passed else 'SOME TESTS FAILED'}")
