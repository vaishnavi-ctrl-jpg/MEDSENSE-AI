---
title: MedSense AI
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
<<<<<<< HEAD
<div align="center">

# 🏥 MedSense AI

### *"Who do I treat first?"*
#### A reinforcement learning agent that learns to answer emergency medicine's most critical question.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-DQN%20%2B%20PPO-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-4CAF50)](https://gymnasium.farama.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-7C3AED)](openenv.yaml)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)](backend/api.py)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-22C55E?logo=pytest&logoColor=white)](tests/)
[![POMDP](https://img.shields.io/badge/POMDP-Noisy%20Vitals-F59E0B)](medsense/noise.py)
[![Domain](https://img.shields.io/badge/Domain-Emergency%20Medicine-EF4444)](medsense/clinical_references.py)
[![Algorithms](https://img.shields.io/badge/RL-DQN%20%2B%20PPO%20%2B%20Rule--Based-8B5CF6)](agents/)
[![Tasks](https://img.shields.io/badge/Tasks-3%20Difficulty%20Levels-06B6D4)](tasks/)
[![Lines](https://img.shields.io/badge/Python-2675%20Lines-64748B)](medsense/)
[![License](https://img.shields.io/badge/License-MIT-6B7280)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Scaler%20%C3%97%20Meta%20%C3%97%20PyTorch-Hackathon-00E5FF)](https://scaler.com)
[![GitHub](https://img.shields.io/badge/GitHub-MEDSENSE--AI-181717?logo=github&logoColor=white)](https://github.com/vaishnavi-ctrl-jpg/MEDSENSE-AI)

</div>

---

## The Problem

Every year, **triage errors in emergency departments cause preventable deaths.**

A nurse arrives at a patient. 30 seconds. Chest pain, elevated BP, cardiac history — is this a heart attack or indigestion? Delay the wrong patient and they could die. Over-prioritise the wrong one and critical resources are wasted.

**MedSense AI trains a reinforcement learning agent to make that decision** — correctly, consistently, with a plain-English explanation for every choice.

---

## What Makes This Research-Grade

| What we built | Why it matters |
|---|---|
| **RL environment** — not a classifier | Agent learns by acting and seeing consequences |
| **POMDP** — Gaussian noise on vitals | Real monitors have ±3% error. Agent never sees perfect data |
| **3-tier open benchmark** — Easy → Hard | Hard mode (5 patients) is unsolved by any heuristic baseline |
| **Two RL algorithms** — DQN + PPO | Demonstrates learnability and algorithm comparison |
| **Clinical grounding** — WHO, AHA, NEWS2 | Every threshold from published medical guidelines |
| **Explainability** — plain-English reasoning | Doctors see *why* the agent chose what it chose |
| **Human-in-the-loop** — doctor override built in | Dashboard shows reasoning; doctors can override |

---

## ⚡ Quickstart & Deployment

### 🌐 Live Demo & HF Space
Our interactive dashboard and model benchmark results are deployed live!
👉 **[Insert your Live Dashboard Link Here]**

### 🐳 Running with Docker (Recommended)
Our environment is fully containerized for reproducibility as per hackathon specs.
```bash
# Build the image
docker build -t medsense-api .

# Run the container
docker run -p 8000:8000 medsense-api
```

---

## 🎯 How It Works

```
Patient arrives in ED
        │
        ▼
Agent observes vitals (with Gaussian noise — POMDP)
  BP: 195/105 mmHg  ·  HR: 118 bpm  ·  SpO2: 89%  ·  Temp: 38.5°C
        │
        ▼
Agent selects action
  [ TREAT NOW ]     [ DELAY ]     [ REFER ]
        │
        ▼
Agent explains its reasoning
  "SpO2 89% (critical hypoxia) + BP 195 (hypertensive crisis)
   + cardiac history → HIGH ACUITY. Required: TREAT NOW."
        │
        ▼
Reward signal reinforces the right behaviour
  Correct critical patient identified → +2.0
  Missed critical patient             → -2.0  (hard clinical limit: <5% miss rate)
```
## 🔌 OpenEnv Interface & Specs

Our project fully implements the OpenEnv standard for reproducible RL evaluation.

### 1. Standard Interface
* **`reset()`**: Initializes the ED environment and generates a stochastic patient queue.
* **`step(action)`**: Advances the environment by one decision step. Returns `(observation, reward, terminated, truncated, info)`.
* **`state()`**: Returns the complete internal state of the ED.

### 2. Action & Observation Space
* **Action Space (Discrete: 3):** The wrapper automatically converts LLM text (e.g., "Treat patient 2") into these structured actions:
  * `0`: **TREAT NOW** (Immediate ED Admission)
  * `1`: **DELAY** (Place in Waiting Area)
  * `2`: **REFER** (To Urgent Care / Clinic)
* **Observation Space:** A structured vector `[age, sys_bp, dia_bp, hr, spo2, temp, complaint_encoded]`.

### 3. Reward Function
Our shaped reward system heavily penalizes clinical mistakes while rewarding efficiency:
| Outcome | Reward | Description |
|---|---|---|
| **Correct Action** | `+1.0` | Decision perfectly matches clinical guidelines. |
| **Critical Miss** | `-2.0` | ⚠️ Delaying or referring a CRITICAL patient. |
| **Over-triage** | `-0.5` | Treating a stable patient, wasting valuable ED beds. |
| **Wrong Queue** | `-0.3` | Incorrectly referring an urgent patient. |
---

## 📊 Benchmark Results *(verified — 50 episodes, seed=42)*

| Agent | Task | Accuracy | Critical Miss | Result |
|---|---|---|---|---|
| **Rule-Based** | Easy | **94%** | **0%** | ✅ PASS |
| **Rule-Based** | Medium | **86%** | **0%** | ✅ PASS |
| **Rule-Based** | Hard | **86%** | **0%** | ✅ PASS |
| Random | Easy | 28% | 30% | ❌ FAIL |
| Random | Medium | 32% | 24% | ❌ FAIL |
| Random | **Hard** | **23%** | **74%** | ❌ FAIL |

> Random agent misses **74% of critical patients** on Hard mode — proving the task is genuinely hard and non-trivial.
> Rule-based passes all 3 tasks with 0% critical miss — proving the environment is correctly calibrated.
> DQN and PPO agents produce improving reward curves — proving the environment is learnable for neural agents.

---

## 🧪 3 Tasks Explained

### Task 1 — Easy: The Textbook Case
```
Patients: 1  ·  Noise: 5%  ·  Threshold: 85% accuracy
```
One patient with clearly dangerous or clearly normal vitals. SpO2 84% = treat now. All normal = delay. Designed to verify the agent learns basic clinical rules.

### Task 2 — Medium: The Ambiguous Case
```
Patients: 1  ·  Noise: 10%  ·  Comorbidities: yes  ·  Threshold: 72% accuracy
```
One patient with borderline vitals and medical history. HR 108 is fine for a healthy 25-year-old — dangerous for a 67-year-old with cardiac history. Agent must use context.

### Task 3 — Hard: The Queue *(Open Benchmark)*
```
Patients: 5  ·  Noise: 15%  ·  Time pressure  ·  Threshold: 60% + <5% critical miss
```
Five patients simultaneously. Agent must prioritise the queue. Delay a critical patient and the episode terminates early with a large penalty. **No heuristic fully solves this.**

---

## 🔬 Clinical Grounding

Every threshold is derived from published medical literature — not made up.

| Threshold | Source |
|---|---|
| SpO2 < 90% = critical hypoxia | WHO Pulse Oximetry Training Manual (2011) |
| SBP > 180 = hypertensive crisis | AHA/ACC Hypertension Guidelines (2017) |
| HR > 130 = dangerous tachycardia | ACLS Guidelines — American Heart Association |
| RR > 25 = urgent respiratory | NEWS2 Score — Royal College of Physicians UK (2017) |
| 3-class triage (treat/delay/refer) | Manchester Triage System (MTS) |
| Measurement noise model | Jubran (2015); Pickering et al. (2005) |

---

## 🏗️ Project Structure

```
MEDSENSE-AI/
│
├── 📄 openenv.yaml              OpenEnv compliance spec (hackathon requirement)
├── 📄 requirements.txt          Dependencies (no flask_cors needed)
├── 📄 start.py                  One-command launcher
├── 📄 run_demo.py               Terminal demo
│
├── 🧠 medsense/                 Core RL Environment
│   ├── triage_env.py            MedSenseEnv — Gymnasium + OpenEnv
│   ├── patient_generator.py     Synthetic CRITICAL / URGENT / STABLE patients
│   ├── noise.py                 POMDP: Gaussian noise on all vital signs
│   ├── reward.py                Asymmetric clinical reward function
│   ├── grader.py                Hackathon-style task graders
│   ├── models.py                Typed dataclasses (PatientObservation etc.)
│   ├── constants.py             Shared thresholds + OBS_IDX map
│   ├── clinical_references.py   WHO / AHA / NEWS2 / MTS grounding
│   └── scalability.py           Architecture + 4-phase roadmap
│
├── 🤖 agents/
│   ├── rule_based_agent.py      Clinical rules baseline (human benchmark)
│   ├── train.py                 DQN with experience replay
│   ├── ppo_agent.py             PPO Actor-Critic (advanced RL)
│   └── compare_agents.py        Multi-algorithm comparison + JSON output
│
├── 🌐 backend/
│   └── api.py                   Flask REST API v2.0 (10 endpoints)
│
├── 💻 frontend/
│   └── index.html               Live interactive dashboard
│
├── ⚙️  tasks/
│   ├── triage_easy.yaml         Task 1 — Easy config
│   ├── triage_medium.yaml       Task 2 — Medium config
│   └── triage_hard.yaml         Task 3 — Hard config
│
├── 📊 results/                  Comparison output (auto-generated)
├── 📚 docs/
│   └── architecture.md          System architecture diagrams
└── 🧪 tests/
    └── test_medsense.py         24 tests — all passing
```

---

## 🔌 REST API v2.0

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check + version |
| `/api/tasks` | GET | List all 3 tasks with metadata |
| `/api/algorithms` | GET | List available trained models |
| `/api/reset` | POST | Start new episode |
| `/api/step` | POST | Take one action |
| `/api/grade` | POST | Run grader on one task |
| `/api/grade/all` | POST | Run grader on all 3 tasks |
| `/api/leaderboard` | GET | Compare agents |
| `/api/results` | GET | Serve comparison JSON |
| `/api/clinical` | GET | Clinical grounding info |

---

## 🤖 Training

```bash
# DQN (experience replay, 14-dim state, discrete 3-action)
python agents/train.py triage_easy
python agents/train.py triage_hard --episodes 2000

# PPO (Actor-Critic, GAE, clipped surrogate — better stability)
python agents/ppo_agent.py triage_easy
python agents/ppo_agent.py triage_hard --episodes 2000

# Compare all algorithms — saves results/comparison_results.json
python agents/compare_agents.py --episodes 100
```

---

## 📈 Scalability Roadmap

| Phase | Status | What |
|---|---|---|
| 1 — Research | ✅ Complete | Synthetic POMDP, DQN + PPO, API, dashboard |
| 2 — Data | 🔲 Next sprint | MIMIC-IV real patient data integration |
| 3 — Production | 🔲 Future | Docker + TorchServe + horizontal scaling |
| 4 — Clinical | 🔲 Future | Human-in-loop feedback, confidence thresholds, HIPAA logging |

---

## 🧪 Tests

```bash
pytest tests/ -v
# ✅ 24 tests — all passing
```

---

## 👥 Team

| Member | Contribution |
|---|---|
| **ViVi** | RL environment, OpenEnv compliance, patient generator, grader, task YAMLs, PPO agent, clinical grounding, scalability docs |
| **Nidz** | Flask REST API v2.0, DQN training, rule-based agent, multi-algorithm comparison |
| **Pooja** | Live interactive dashboard, README, submission documentation |

---

<div align="center">

**Built for the Scaler × Meta × PyTorch Hackathon 🏆**

*"The goal was not to build AI for healthcare.*
*The goal was to build healthcare-grade AI."*

<br/>

[![View on GitHub](https://img.shields.io/badge/View%20on%20GitHub-vaishnavi--ctrl--jpg%2FMEDSENSE--AI-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vaishnavi-ctrl-jpg/MEDSENSE-AI)

</div>
=======
---
title: MEDSENSE AI
emoji: 🏆
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> bcc7be4d2ce2bf15b91ccf78be2e18704473b132
