"""
MedSense AI — Constants
Shared across grader, agents, and environment.
"""

# ── Action space ──────────────────────────────────────────────────────────────
TREAT_NOW = 0
DELAY     = 1
REFER     = 2

# ── Severity levels — MUST MATCH models.py ───────────────────────────────────
# BUG FIXED: Nidz had CRITICAL=0, STABLE=1, URGENT=2
# ViVi's models.py uses strings: "CRITICAL", "URGENT", "STABLE"
# These int versions are only used inside constants/rule_based_agent
CRITICAL = 0
URGENT   = 1
STABLE   = 2

# ── Clinical thresholds (used by rule_based_agent) ────────────────────────────
THRESHOLDS = {
    "spo2_min":    90.0,   # SpO2 below 90% = critical hypoxia
    "sys_bp_max":  180.0,  # Hypertensive crisis
    "sys_bp_min":  90.0,   # Shock threshold
    "hr_max":      130.0,  # Dangerous tachycardia
    "hr_min":      40.0,   # Dangerous bradycardia
    "temp_max":    39.5,   # High fever threshold
}

# ── Observation vector index map ──────────────────────────────────────────────
# Matches PatientObservation.to_array() order in models.py
OBS_IDX = {
    "bp_systolic":   0,
    "bp_diastolic":  1,
    "heart_rate":    2,
    "spo2":          3,
    "temperature":   4,
    "resp_rate":     5,
    "complaint":     6,
    "pain_score":    7,
    "age":           8,
    "cardiac":       9,
    "diabetes":      10,
    "resp_disease":  11,
    "queue_length":  12,
    "time_waiting":  13,
}
