"""
MedSense AI — Clinical Grounding & References
==============================================
This file documents the real-world clinical basis for MedSense AI's
thresholds, reward design, and task definitions.

All vital sign thresholds are derived from published clinical guidelines.
This is NOT a medical device — it is a research simulation environment.
"""

# ── Clinical threshold references ─────────────────────────────────────────────
CLINICAL_REFERENCES = {

    "spo2_critical": {
        "threshold":  "SpO2 < 90%",
        "source":     "WHO Pulse Oximetry Training Manual (2011)",
        "basis":      "SpO2 below 90% indicates severe hypoxia requiring immediate intervention",
        "action":     "treat_now",
    },

    "hypertensive_crisis": {
        "threshold":  "Systolic BP > 180 mmHg",
        "source":     "AHA/ACC Hypertension Guidelines 2017 (Whelton et al.)",
        "basis":      "Systolic > 180 with symptoms = hypertensive emergency requiring immediate care",
        "action":     "treat_now",
    },

    "shock": {
        "threshold":  "Systolic BP < 90 mmHg",
        "source":     "Surviving Sepsis Campaign Guidelines 2021",
        "basis":      "SBP < 90 mmHg indicates haemodynamic instability / shock state",
        "action":     "treat_now",
    },

    "tachycardia_severe": {
        "threshold":  "HR > 130 bpm",
        "source":     "ACLS Guidelines — American Heart Association",
        "basis":      "HR > 130 with symptoms indicates haemodynamic compromise",
        "action":     "treat_now",
    },

    "bradycardia_severe": {
        "threshold":  "HR < 40 bpm",
        "source":     "ACLS Guidelines — American Heart Association",
        "basis":      "HR < 40 = severe bradycardia risking cardiac arrest",
        "action":     "treat_now",
    },

    "hyperpyrexia": {
        "threshold":  "Temperature > 39.5°C",
        "source":     "NICE Guidelines — Fever in Adults (2024)",
        "basis":      "Temperature > 39.5°C with other symptoms indicates serious infection",
        "action":     "treat_now",
    },

    "tachypnea": {
        "threshold":  "Respiratory Rate > 25 /min",
        "source":     "NEWS2 Score — Royal College of Physicians UK (2017)",
        "basis":      "RR > 25 scores 3 on NEWS2, triggering urgent clinical response",
        "action":     "treat_now",
    },

    "cardiac_comorbidity": {
        "threshold":  "Cardiac history + HR > 110 OR SBP > 155",
        "source":     "ESC Heart Failure Guidelines 2021",
        "basis":      "Patients with prior cardiac disease have lower decompensation threshold",
        "action":     "treat_now",
    },
}

# ── Triage system basis ───────────────────────────────────────────────────────
TRIAGE_SYSTEM_BASIS = {
    "system":   "Manchester Triage System (MTS) + NEWS2 Score",
    "sources": [
        "Mackway-Jones K et al. — Emergency Triage: Manchester Triage Group (3rd Ed)",
        "Royal College of Physicians — National Early Warning Score 2 (NEWS2) 2017",
        "Australasian College for Emergency Medicine — Guidelines on the Implementation of the Australasian Triage Scale",
    ],
    "description": (
        "MedSense AI's 3-class action space (treat_now, delay, refer) maps to "
        "the Manchester Triage System categories 1-2 (immediate/very urgent), "
        "3-4 (urgent/standard), and 5 (non-urgent / referral pathway). "
        "Vital sign thresholds align with NEWS2 score trigger points."
    ),
}

# ── POMDP justification ───────────────────────────────────────────────────────
POMDP_CLINICAL_BASIS = {
    "description": (
        "Clinical monitors have documented measurement error. "
        "Pulse oximetry has ±2-3% accuracy in ideal conditions, worse in motion, "
        "poor perfusion, or dark skin tones. Blood pressure cuffs show ±5-10 mmHg "
        "variance. MedSense AI models this as Gaussian noise on all vital signs, "
        "creating a Partially Observable Markov Decision Process (POMDP) that "
        "reflects the true uncertainty clinicians face."
    ),
    "sources": [
        "Jubran A — Pulse oximetry. Crit Care 2015",
        "Pickering TG et al. — Recommendations for Blood Pressure Measurement. Hypertension 2005",
    ],
}

# ── Reward function clinical basis ────────────────────────────────────────────
REWARD_CLINICAL_BASIS = {
    "missed_critical": {
        "reward":  -2.0,
        "basis":   "Missing a critical patient is the highest-risk error in ED triage. "
                   "Under-triage of 1% increases mortality by up to 10% (Considine et al.)",
        "source":  "Considine J et al. — Triage and patient outcomes. EMA 2016",
    },
    "over_triage": {
        "reward":  -0.5,
        "basis":   "Over-triage wastes ED resources but is safer than under-triage. "
                   "Penalty is lower to reflect asymmetric clinical risk.",
        "source":  "Australasian College for Emergency Medicine — ATS Guidelines",
    },
    "correct_critical": {
        "reward":  +2.0,
        "basis":   "Correctly identifying a critical patient enables timely intervention, "
                   "directly reducing time-to-treatment and mortality.",
    },
}


def print_summary():
    print("\n" + "="*60)
    print("  MedSense AI — Clinical Grounding Summary")
    print("="*60)
    print(f"\n  Triage basis: {TRIAGE_SYSTEM_BASIS['system']}")
    print(f"\n  Thresholds grounded in {len(CLINICAL_REFERENCES)} clinical guidelines")
    print(f"  POMDP noise justified by clinical measurement literature")
    print(f"  Reward asymmetry reflects real clinical risk hierarchy")
    print("\n  Key sources:")
    for s in TRIAGE_SYSTEM_BASIS["sources"]:
        print(f"    • {s}")
    for s in POMDP_CLINICAL_BASIS["sources"]:
        print(f"    • {s}")
    print()


if __name__ == "__main__":
    print_summary()
    print("  Threshold details:")
    for name, ref in CLINICAL_REFERENCES.items():
        print(f"\n  {name.upper()}")
        print(f"    Threshold: {ref['threshold']}")
        print(f"    Source:    {ref['source']}")
        print(f"    Basis:     {ref['basis']}")
