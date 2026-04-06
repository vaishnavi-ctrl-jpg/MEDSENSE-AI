"""
MedSense AI — Patient Generator
Generates realistic synthetic patients for all 3 task difficulties.

Each patient has:
  - Vital signs (true values, before noise)
  - Chief complaint
  - Medical history
  - Ground truth severity (CRITICAL / URGENT / STABLE)
  - Correct action (treat_now / delay / refer)
  - Clinical reasoning string
"""

import random
from typing import List, Optional

from .models import (
    PatientObservation, PatientRecord,
    SEVERITY_CRITICAL, SEVERITY_URGENT, SEVERITY_STABLE,
    ACTION_TREAT_NOW, ACTION_DELAY, ACTION_REFER,
    COMPLAINTS, HIGH_ACUITY_COMPLAINTS, SPECIALIST_COMPLAINTS,
)

# ── Patient name pool ─────────────────────────────────────────────────────────
FIRST_NAMES_F = ["Sarah", "Emma", "Lisa", "Priya", "Maria", "Aisha", "Chen", "Fatima"]
FIRST_NAMES_M = ["David", "James", "Omar", "Raj",   "Marco", "Ahmed", "Liam", "Kenji"]
LAST_INITIALS = ["J.", "K.", "R.", "T.", "M.", "S.", "P.", "L.", "A.", "B."]


class PatientGenerator:
    """
    Generates synthetic patients for MedSense triage episodes.

    Severity distribution per difficulty:
        Easy:   40% Critical, 30% Urgent, 30% Stable
        Medium: 30% Critical, 40% Urgent, 30% Stable (more ambiguous)
        Hard:   35% Critical, 40% Urgent, 25% Stable (harder to classify)
    """

    SEVERITY_WEIGHTS = {
        "easy":   [0.40, 0.30, 0.30],
        "medium": [0.30, 0.40, 0.30],
        "hard":   [0.35, 0.40, 0.25],
    }

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        self.difficulty = difficulty
        self._rng = random.Random(seed)

    def generate(self) -> PatientRecord:
        """Generate one complete patient record."""
        weights  = self.SEVERITY_WEIGHTS.get(self.difficulty, [0.33, 0.33, 0.34])
        severity = self._rng.choices(
            [SEVERITY_CRITICAL, SEVERITY_URGENT, SEVERITY_STABLE],
            weights=weights,
            k=1
        )[0]

        if severity == SEVERITY_CRITICAL:
            return self._critical_patient()
        elif severity == SEVERITY_URGENT:
            return self._urgent_patient()
        else:
            return self._stable_patient()

    def generate_queue(self, n: int) -> List[PatientRecord]:
        """Generate a queue of n patients (used in Hard mode)."""
        return [self.generate() for _ in range(n)]

    # ── Patient builders ──────────────────────────────────────────────────────

    def _critical_patient(self) -> PatientRecord:
        """
        CRITICAL: Life-threatening presentation.
        Always requires treat_now.
        Vitals: at least one danger threshold breached.
        """
        age    = self._rng.randint(35, 85)
        gender = self._rng.choice(["M", "F"])
        name   = self._random_name(gender)

        # Pick a danger pattern
        pattern = self._rng.choice([
            "respiratory_failure",
            "hypertensive_crisis",
            "cardiac_event",
            "shock",
        ])

        if pattern == "respiratory_failure":
            obs = PatientObservation(
                bp_systolic   = self._rng.uniform(90, 130),
                bp_diastolic  = self._rng.uniform(60, 85),
                heart_rate    = self._rng.uniform(100, 135),
                spo2          = self._rng.uniform(82, 91),    # DANGER: < 92%
                temperature   = self._rng.uniform(36.5, 38.5),
                resp_rate     = self._rng.uniform(22, 32),    # DANGER: > 20
                chief_complaint = 1,   # Shortness of Breath
                pain_score    = self._rng.randint(5, 9),
                age           = age,
                has_respiratory_disease = 1,
            )
            reasoning = (
                f"SpO2 {obs.spo2:.0f}% (critical hypoxia) + "
                f"resp rate {obs.resp_rate:.0f} (tachypnoea) → CRITICAL"
            )

        elif pattern == "hypertensive_crisis":
            obs = PatientObservation(
                bp_systolic   = self._rng.uniform(185, 220),  # DANGER: > 180
                bp_diastolic  = self._rng.uniform(110, 135),
                heart_rate    = self._rng.uniform(90, 120),
                spo2          = self._rng.uniform(93, 97),
                temperature   = self._rng.uniform(36.5, 37.5),
                resp_rate     = self._rng.uniform(16, 22),
                chief_complaint = 0,   # Chest Pain
                pain_score    = self._rng.randint(7, 10),
                age           = age,
                has_cardiac_history = 1,
                has_diabetes  = self._rng.choice([0, 1]),
            )
            reasoning = (
                f"BP {obs.bp_systolic:.0f}/{obs.bp_diastolic:.0f} (hypertensive crisis) + "
                f"chest pain + cardiac history → CRITICAL"
            )

        elif pattern == "cardiac_event":
            obs = PatientObservation(
                bp_systolic   = self._rng.uniform(80, 115),   # low — cardiogenic shock risk
                bp_diastolic  = self._rng.uniform(50, 75),
                heart_rate    = self._rng.uniform(110, 140),  # DANGER: tachycardia
                spo2          = self._rng.uniform(88, 93),
                temperature   = self._rng.uniform(36.0, 37.5),
                resp_rate     = self._rng.uniform(18, 26),
                chief_complaint = 0,   # Chest Pain
                pain_score    = self._rng.randint(8, 10),
                age           = age,
                has_cardiac_history = 1,
            )
            reasoning = (
                f"HR {obs.heart_rate:.0f}bpm (tachycardia) + "
                f"SpO2 {obs.spo2:.0f}% + chest pain + cardiac history → CRITICAL"
            )

        else:  # shock
            obs = PatientObservation(
                bp_systolic   = self._rng.uniform(70, 88),    # DANGER: < 90 = shock
                bp_diastolic  = self._rng.uniform(40, 60),
                heart_rate    = self._rng.uniform(115, 145),
                spo2          = self._rng.uniform(90, 95),
                temperature   = self._rng.uniform(35.0, 36.5),
                resp_rate     = self._rng.uniform(20, 30),
                chief_complaint = self._rng.choice([0, 3]),
                pain_score    = self._rng.randint(6, 10),
                age           = age,
            )
            reasoning = (
                f"BP {obs.bp_systolic:.0f}/{obs.bp_diastolic:.0f} (shock) + "
                f"HR {obs.heart_rate:.0f}bpm → CRITICAL"
            )

        return PatientRecord(
            name=name, age=age, gender=gender,
            observation=obs,
            true_severity=SEVERITY_CRITICAL,
            correct_action=ACTION_TREAT_NOW,
            reasoning=reasoning,
            is_ambiguous=False,
        )

    def _urgent_patient(self) -> PatientRecord:
        """
        URGENT: Significant presentation, needs timely attention.
        Correct action depends on complaint and context.
        In medium/hard mode, may be genuinely ambiguous.
        """
        age    = self._rng.randint(20, 75)
        gender = self._rng.choice(["M", "F"])
        name   = self._random_name(gender)
        ambiguous = (self.difficulty in ["medium", "hard"] and
                     self._rng.random() < 0.4)

        complaint = self._rng.choice([0, 1, 2, 5])

        obs = PatientObservation(
            bp_systolic   = self._rng.uniform(130, 160),
            bp_diastolic  = self._rng.uniform(85, 105),
            heart_rate    = self._rng.uniform(90, 115),
            spo2          = self._rng.uniform(93, 97),
            temperature   = self._rng.uniform(37.5, 39.0),
            resp_rate     = self._rng.uniform(18, 24),
            chief_complaint = complaint,
            pain_score    = self._rng.randint(4, 7),
            age           = age,
            has_cardiac_history = 1 if ambiguous else 0,
            has_diabetes  = self._rng.choice([0, 1]),
        )

        # Urgent patients with cardiac history + borderline vitals → treat_now
        if ambiguous and obs.has_cardiac_history and obs.heart_rate > 105:
            correct_action = ACTION_TREAT_NOW
            reasoning = (
                f"Borderline HR {obs.heart_rate:.0f} + cardiac history → "
                f"elevated risk, treat_now"
            )
        elif complaint in SPECIALIST_COMPLAINTS:
            correct_action = ACTION_REFER
            reasoning = f"Complaint requires specialist referral"
        else:
            correct_action = ACTION_TREAT_NOW
            reasoning = (
                f"Elevated BP {obs.bp_systolic:.0f}, HR {obs.heart_rate:.0f}, "
                f"temp {obs.temperature:.1f}°C → urgent care needed"
            )

        return PatientRecord(
            name=name, age=age, gender=gender,
            observation=obs,
            true_severity=SEVERITY_URGENT,
            correct_action=correct_action,
            reasoning=reasoning,
            is_ambiguous=ambiguous,
        )

    def _stable_patient(self) -> PatientRecord:
        """
        STABLE: Non-urgent presentation.
        Usually delay, sometimes refer for specialist care.
        """
        age    = self._rng.randint(18, 65)
        gender = self._rng.choice(["M", "F"])
        name   = self._random_name(gender)

        complaint = self._rng.choice([4, 7, 8, 9])  # minor complaints

        obs = PatientObservation(
            bp_systolic   = self._rng.uniform(100, 135),
            bp_diastolic  = self._rng.uniform(65, 85),
            heart_rate    = self._rng.uniform(62, 95),
            spo2          = self._rng.uniform(96, 100),
            temperature   = self._rng.uniform(36.1, 37.5),
            resp_rate     = self._rng.uniform(12, 18),
            chief_complaint = complaint,
            pain_score    = self._rng.randint(1, 4),
            age           = age,
        )

        if complaint in SPECIALIST_COMPLAINTS:
            correct_action = ACTION_REFER
            reasoning = f"{COMPLAINTS[complaint]} → specialist referral appropriate"
        else:
            correct_action = ACTION_DELAY
            reasoning = (
                f"All vitals within normal range, minor complaint "
                f"({COMPLAINTS[complaint]}) → safe to delay"
            )

        return PatientRecord(
            name=name, age=age, gender=gender,
            observation=obs,
            true_severity=SEVERITY_STABLE,
            correct_action=correct_action,
            reasoning=reasoning,
            is_ambiguous=False,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _random_name(self, gender: str) -> str:
        if gender == "F":
            first = self._rng.choice(FIRST_NAMES_F)
        else:
            first = self._rng.choice(FIRST_NAMES_M)
        last = self._rng.choice(LAST_INITIALS)
        return f"{first} {last}"
