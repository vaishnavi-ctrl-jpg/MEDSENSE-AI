# MedSense AI — Architecture

## System Data Flow

```
Patient Generator ──▶ POMDP Noise ──▶ Agent (DQN/PPO/Rule-Based)
                                              │
                                              ▼ action
                                       MedSenseEnv.step()
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                           Reward         GradeResult    EpisodeReport
                           Signal         (correct,      (accuracy,
                                          critical_miss) crit_miss_rate)
                              │
                              ▼
                       Flask REST API
                              │ JSON
                              ▼
                       Live Dashboard
```

## Task Difficulty

```
EASY (85%)      MEDIUM (72%)     HARD (60% + <5% miss)
──────────────  ───────────────  ─────────────────────
1 patient       1 patient        5 patients
5% noise        10% noise        15% noise
Clear vitals    Borderline       High ambiguity
No history      Comorbidities    Time pressure
Rule-based: ✅  Rule-based: ✅   Open benchmark
```

## POMDP

```
True vitals (hidden) ──▶ Gaussian noise ──▶ Agent sees noisy observation
                         σ = 5/10/15%        (never true values)
                                              │
Ground truth only ◀───────────────────────── Reward computed here
used for reward
```
