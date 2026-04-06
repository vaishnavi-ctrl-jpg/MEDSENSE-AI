"""
MedSense AI — Reward Calculator
Clinical reward function aligned with the openenv.yaml spec.

Key design:
  - Missing a critical patient is the worst outcome (-2.0)
  - Correct decisions on ambiguous cases get bonus (+2.0)
  - Asymmetric penalties: false negatives > false positives
"""

from .models import (
    PatientRecord, GradeResult,
    SEVERITY_CRITICAL, SEVERITY_URGENT, SEVERITY_STABLE,
    ACTION_TREAT_NOW, ACTION_DELAY, ACTION_REFER,
    ACTION_NAMES,
)


class RewardCalculator:
    """
    Computes reward for one triage decision.
    Returns a GradeResult with reward + detailed explanation.
    """

    # ── Reward table (matches openenv.yaml) ───────────────────────────────────
    R_CORRECT_PRIORITY = +1.0
    R_CORRECT_ACTION   = +1.0
    R_AMBIGUOUS_BONUS  = +2.0
    R_MISSED_CRITICAL  = -1.0
    R_OVER_TRIAGE      = -0.5
    R_WRONG_REFER      = -0.3
    R_TIMEOUT          = -0.2
    R_DETERIORATED     = -2.0

    def compute(self, action: int, patient: PatientRecord) -> GradeResult:
        """Grade one triage decision against ground truth."""
        ground_truth   = patient.correct_action
        severity       = patient.true_severity
        correct        = (action == ground_truth)

        # ── Flag error types ──────────────────────────────────────────────────
        critical_miss = (
            severity == SEVERITY_CRITICAL and
            action != ACTION_TREAT_NOW
        )
        over_triage = (
            severity == SEVERITY_STABLE and
            action == ACTION_TREAT_NOW
        )
        wrong_refer = (
            ground_truth == ACTION_TREAT_NOW and
            action == ACTION_REFER
        )

        # ── Compute reward ────────────────────────────────────────────────────
        reward = 0.0
        reasons = []

        if correct:
            reward += self.R_CORRECT_PRIORITY
            reward += self.R_CORRECT_ACTION
            reasons.append(f"Correct priority (+{self.R_CORRECT_PRIORITY + self.R_CORRECT_ACTION})")

            if patient.is_ambiguous:
                reward += self.R_AMBIGUOUS_BONUS
                reasons.append(f"Ambiguous case bonus (+{self.R_AMBIGUOUS_BONUS})")
        else:
            if critical_miss:
                reward += self.R_MISSED_CRITICAL
                # Extra penalty if patient would deteriorate
                if severity == SEVERITY_CRITICAL:
                    reward += self.R_DETERIORATED
                    reasons.append(f"Critical patient missed → deterioration risk ({self.R_MISSED_CRITICAL + self.R_DETERIORATED})")
                else:
                    reasons.append(f"Critical patient missed ({self.R_MISSED_CRITICAL})")

            elif over_triage:
                reward += self.R_OVER_TRIAGE
                reasons.append(f"Over-triaged stable patient ({self.R_OVER_TRIAGE})")

            elif wrong_refer:
                reward += self.R_WRONG_REFER
                reasons.append(f"Referred when treat_now needed ({self.R_WRONG_REFER})")

            else:
                # General incorrect decision
                reward += self.R_MISSED_CRITICAL * 0.5
                reasons.append(f"Incorrect decision ({self.R_MISSED_CRITICAL * 0.5})")

        explanation = (
            f"Patient: {patient.name} ({severity}) | "
            f"Agent: {ACTION_NAMES[action]} | "
            f"Correct: {ACTION_NAMES[ground_truth]} | "
            + " | ".join(reasons)
        )

        return GradeResult(
            correct        = correct,
            critical_miss  = critical_miss,
            over_triage    = over_triage,
            wrong_refer    = wrong_refer,
            reward         = round(reward, 2),
            ground_truth   = ACTION_NAMES[ground_truth],
            agent_action   = ACTION_NAMES[action],
            patient_name   = patient.name,
            severity       = severity,
            explanation    = explanation,
        )
