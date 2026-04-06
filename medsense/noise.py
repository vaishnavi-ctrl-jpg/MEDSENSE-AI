"""
MedSense AI — POMDP Noise Model
Applies realistic Gaussian noise to patient vital signs.

This is what makes MedSense a POMDP (Partially Observable MDP):
the agent never sees true vital signs — only noisy measurements,
just like a real clinical monitor.

Noise levels by difficulty:
  Easy:   σ = 0.05 (very small noise — mostly accurate readings)
  Medium: σ = 0.10 (moderate noise — some ambiguity)
  Hard:   σ = 0.15 (high noise — real measurement uncertainty)
"""

import random
from dataclasses import replace
from typing import Optional

from .models import PatientObservation


# ── Noise standard deviations per vital sign ──────────────────────────────────
# Expressed as fraction of the normal range span.
# e.g. BP systolic range ≈ 80 mmHg → σ=0.10 → ~8 mmHg noise
VITAL_NOISE_FRACTIONS = {
    "bp_systolic":   80,    # range width ≈ 80 mmHg
    "bp_diastolic":  40,    # range width ≈ 40 mmHg
    "heart_rate":    80,    # range width ≈ 80 bpm
    "spo2":          10,    # range width ≈ 10 %
    "temperature":   3,     # range width ≈ 3 °C
    "resp_rate":     16,    # range width ≈ 16 breaths/min
}

DIFFICULTY_SIGMA = {
    "easy":   0.05,
    "normal": 0.10,
    "medium": 0.10,
    "hard":   0.15,
}


class VitalNoiseModel:
    """
    Applies per-vital Gaussian noise to a PatientObservation.

    Usage:
        model = VitalNoiseModel(difficulty="hard")
        noisy_obs = model.apply(true_observation)
    """

    def __init__(self, difficulty: str = "normal", seed: Optional[int] = None):
        self.sigma = DIFFICULTY_SIGMA.get(difficulty, 0.10)
        self._rng  = random.Random(seed)

    def apply(self, obs: PatientObservation) -> PatientObservation:
        """Return a new observation with noise applied to vital signs."""
        return replace(
            obs,
            bp_systolic  = self._noisy(obs.bp_systolic,  "bp_systolic"),
            bp_diastolic = self._noisy(obs.bp_diastolic, "bp_diastolic"),
            heart_rate   = self._noisy(obs.heart_rate,   "heart_rate"),
            spo2         = self._noisy_clipped(obs.spo2, "spo2", lo=50.0, hi=100.0),
            temperature  = self._noisy(obs.temperature,  "temperature"),
            resp_rate    = self._noisy_clipped(obs.resp_rate, "resp_rate", lo=1.0, hi=60.0),
            # categorical fields are NOT noisy
        )

    def _noisy(self, value: float, vital: str) -> float:
        std = VITAL_NOISE_FRACTIONS[vital] * self.sigma
        return round(value + self._rng.gauss(0, std), 1)

    def _noisy_clipped(self, value: float, vital: str, lo: float, hi: float) -> float:
        std  = VITAL_NOISE_FRACTIONS[vital] * self.sigma
        raw  = value + self._rng.gauss(0, std)
        return round(max(lo, min(hi, raw)), 1)

    def noise_level_description(self) -> str:
        return {
            0.05: "Low noise (easy mode)",
            0.10: "Moderate noise (medium mode)",
            0.15: "High noise (hard mode)",
        }.get(self.sigma, f"σ={self.sigma}")
