"""
MedSense AI — Typed Data Models
All structured data types used across the environment.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── Severity constants ────────────────────────────────────────────────────────
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_URGENT   = "URGENT"
SEVERITY_STABLE   = "STABLE"

# ── Action constants ──────────────────────────────────────────────────────────
ACTION_TREAT_NOW = 0
ACTION_DELAY     = 1
ACTION_REFER     = 2

ACTION_NAMES = {
    ACTION_TREAT_NOW: "treat_now",
    ACTION_DELAY:     "delay",
    ACTION_REFER:     "refer",
}

ACTION_FROM_NAME = {v: k for k, v in ACTION_NAMES.items()}

# ── Chief complaint categories ────────────────────────────────────────────────
COMPLAINTS = {
    0: "Chest Pain",
    1: "Shortness of Breath",
    2: "Abdominal Pain",
    3: "Head Injury / Trauma",
    4: "Fever / Infection",
    5: "Dizziness / Syncope",
    6: "Allergic Reaction",
    7: "Fracture / Orthopaedic",
    8: "Mental Health Crisis",
    9: "Minor Laceration / Wound",
}

# Complaints that warrant immediate treatment if combined with danger vitals
HIGH_ACUITY_COMPLAINTS = {0, 1, 3, 5, 6}

# Complaints that lean toward referral
SPECIALIST_COMPLAINTS = {7, 8}


# ── Core data types ───────────────────────────────────────────────────────────

@dataclass
class PatientObservation:
    """
    What the agent observes about a patient.
    Vital signs include Gaussian noise (POMDP).
    Ground truth severity is NOT included here — it's in PatientRecord.
    """
    # Vital signs (continuous, noisy)
    bp_systolic:            float   # mmHg
    bp_diastolic:           float   # mmHg
    heart_rate:             float   # bpm
    spo2:                   float   # %
    temperature:            float   # °C
    resp_rate:              float   # breaths/min

    # Categorical / ordinal
    chief_complaint:        int     # 0–9 (see COMPLAINTS)
    pain_score:             int     # 0–10
    age:                    int     # years

    # Medical history flags (0 or 1)
    has_cardiac_history:        int = 0
    has_diabetes:               int = 0
    has_respiratory_disease:    int = 0

    # Queue context (Hard mode only)
    queue_length:           int = 0
    time_waiting_minutes:   float = 0.0

    def to_array(self):
        """Convert to flat numpy-compatible list for the agent."""
        import numpy as np
        return np.array([
            self.bp_systolic, self.bp_diastolic, self.heart_rate,
            self.spo2, self.temperature, self.resp_rate,
            self.chief_complaint, self.pain_score, self.age,
            self.has_cardiac_history, self.has_diabetes,
            self.has_respiratory_disease,
            self.queue_length, self.time_waiting_minutes,
        ], dtype=np.float32)

    def vital_summary(self) -> str:
        return (
            f"BP {self.bp_systolic:.0f}/{self.bp_diastolic:.0f}, "
            f"HR {self.heart_rate:.0f}bpm, "
            f"SpO2 {self.spo2:.0f}%, "
            f"Temp {self.temperature:.1f}°C"
        )


@dataclass
class PatientRecord:
    """
    Full patient record including ground truth (not seen by agent).
    Used by grader and patient generator.
    """
    name:             str
    age:              int
    gender:           str           # "M" / "F"
    observation:      PatientObservation
    true_severity:    str           # CRITICAL / URGENT / STABLE
    correct_action:   int           # ACTION_TREAT_NOW / DELAY / REFER
    reasoning:        str           # Clinical justification
    is_ambiguous:     bool = False  # True = borderline case


@dataclass
class TriageDecision:
    """Agent's output for one patient."""
    action:       int           # 0/1/2
    action_name:  str           # "treat_now" / "delay" / "refer"
    confidence:   float         # 0.0–1.0 (optional, for explainability)
    reasoning:    str           # Human-readable explanation


@dataclass
class GradeResult:
    """Result of grading one triage decision."""
    correct:            bool
    critical_miss:      bool    # Agent delayed/referred a CRITICAL patient
    over_triage:        bool    # Agent treated a STABLE patient as TREAT_NOW
    wrong_refer:        bool    # Agent referred when treat_now was needed
    reward:             float
    ground_truth:       str     # "treat_now" / "delay" / "refer"
    agent_action:       str
    patient_name:       str
    severity:           str
    explanation:        str     # Why this grade was given


@dataclass
class EpisodeReport:
    """Aggregate results for one full episode."""
    task_id:                str
    total_patients:         int
    correct_decisions:      int
    critical_misses:        int
    over_triages:           int
    total_reward:           float
    triage_accuracy:        float   # correct / total
    critical_miss_rate:     float   # critical_misses / total_critical
    over_triage_rate:       float   # over_triages / total_stable
    decision_log:           List[GradeResult] = field(default_factory=list)
    passed:                 bool = False

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"  {status} | Task: {self.task_id}\n"
            f"  Accuracy:          {self.triage_accuracy*100:.1f}%\n"
            f"  Critical Miss Rate:{self.critical_miss_rate*100:.1f}%\n"
            f"  Over-Triage Rate:  {self.over_triage_rate*100:.1f}%\n"
            f"  Total Reward:      {self.total_reward:.2f}"
        )


@dataclass
class TaskReport:
    """Results from running N episodes on one task."""
    task_id:                str
    n_episodes:             int
    win_rate:               float   # episodes where passed=True
    avg_reward:             float
    avg_accuracy:           float
    avg_critical_miss_rate: float
    avg_over_triage_rate:   float
    std_reward:             float
    task_passed:            bool    # overall pass/fail for this task
    episodes:               List[EpisodeReport] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.task_passed else "❌ FAIL"
        bar = "=" * 50
        return (
            f"\n{bar}\n"
            f"  {status} — Task: {self.task_id}\n"
            f"{bar}\n"
            f"  Episodes:          {self.n_episodes}\n"
            f"  Win Rate:          {self.win_rate*100:.1f}%\n"
            f"  Avg Reward:        {self.avg_reward:.2f} ± {self.std_reward:.2f}\n"
            f"  Avg Accuracy:      {self.avg_accuracy*100:.1f}%\n"
            f"  Critical Miss:     {self.avg_critical_miss_rate*100:.1f}%\n"
            f"  Over-Triage:       {self.avg_over_triage_rate*100:.1f}%\n"
            f"{bar}"
        )
