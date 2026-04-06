"""
MedSense AI — Core Triage Environment
Gymnasium-compatible, OpenEnv-compliant RL environment.

The agent observes patient vitals (with noise) and must decide:
    0 = treat_now  |  1 = delay  |  2 = refer

Episode structure:
    - Each step = one patient
    - Easy/Medium: 1 patient per episode
    - Hard: 5 patients per queue per episode

POMDP: vital signs include Gaussian noise (see noise.py)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict, List

from .models import (
    PatientRecord, GradeResult, EpisodeReport,
    ACTION_TREAT_NOW, ACTION_DELAY, ACTION_REFER, ACTION_NAMES,
    SEVERITY_CRITICAL, SEVERITY_URGENT, SEVERITY_STABLE,
    COMPLAINTS,
)
from .patient_generator import PatientGenerator
from .noise import VitalNoiseModel
from .reward import RewardCalculator


# ── Default task configs (fallback if YAML not found) ─────────────────────────
DEFAULT_CONFIGS = {
    "triage_easy": dict(
        task_id="triage_easy", difficulty="easy",
        patients_per_episode=1, noise_level=0.05,
        ambiguity="low", comorbidities=False,
        queue=False, time_pressure=False,
        success_threshold=0.85, max_steps=20,
    ),
    "triage_medium": dict(
        task_id="triage_medium", difficulty="medium",
        patients_per_episode=1, noise_level=0.10,
        ambiguity="medium", comorbidities=True,
        queue=False, time_pressure=False,
        success_threshold=0.72, max_steps=30,
    ),
    "triage_hard": dict(
        task_id="triage_hard", difficulty="hard",
        patients_per_episode=5, noise_level=0.15,
        ambiguity="high", comorbidities=True,
        queue=True, time_pressure=True,
        success_threshold=0.60, max_steps=50,
    ),
}

OBS_DIM = 14  # number of features in PatientObservation.to_array()


class MedSenseEnv(gym.Env):
    """
    MedSense AI — Hospital Triage RL Environment

    Observation (14-dim float32):
        [bp_systolic, bp_diastolic, heart_rate, spo2, temperature,
         resp_rate, chief_complaint, pain_score, age,
         has_cardiac, has_diabetes, has_resp_disease,
         queue_length, time_waiting]

    Actions (Discrete 3):
        0: treat_now  |  1: delay  |  2: refer

    Tasks:
        triage_easy   — single patient, clear vitals
        triage_medium — single patient, ambiguous + comorbidities
        triage_hard   — 5-patient queue, time pressure
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        task_id:     str = "triage_easy",
        config:      Optional[Dict] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.task_id     = task_id
        self.render_mode = render_mode
        self.cfg         = self._load_config(task_id, config)

        # Gymnasium spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low  = np.zeros(OBS_DIM, dtype=np.float32),
            high = np.array([
                300, 200, 250, 100, 45, 60,   # vitals
                9, 10, 100,                    # complaint, pain, age
                1, 1, 1,                       # history flags
                10, 120,                       # queue_length, time_waiting
            ], dtype=np.float32),
            dtype = np.float32,
        )

        # Sub-modules
        self._generator   = PatientGenerator(self.cfg["difficulty"])
        self._noise_model = VitalNoiseModel(self.cfg["difficulty"])
        self._reward_calc = RewardCalculator()

        # Episode state
        self._patient_queue: List[PatientRecord] = []
        self._current_patient: Optional[PatientRecord] = None
        self._step_idx:  int = 0
        self._episode_log: List[GradeResult] = []
        self._total_reward: float = 0.0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        task_id:  Optional[str] = None,
        seed:     Optional[int] = None,
        options:  Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if task_id and task_id != self.task_id:
            self.task_id = task_id
            self.cfg     = self._load_config(task_id)
            self._generator   = PatientGenerator(self.cfg["difficulty"])
            self._noise_model = VitalNoiseModel(self.cfg["difficulty"])

        if seed is not None:
            self._generator._rng.seed(seed)
            self._noise_model._rng.seed(seed + 1)

        # Generate patient queue for this episode
        n = self.cfg["patients_per_episode"]
        self._patient_queue = self._generator.generate_queue(n)
        self._step_idx     = 0
        self._episode_log  = []
        self._total_reward = 0.0

        # Load first patient
        self._current_patient = self._patient_queue[self._step_idx]
        obs = self._make_obs(self._current_patient)

        info = {
            "task_id":           self.task_id,
            "difficulty":        self.cfg["difficulty"],
            "patients_in_queue": n,
            "patient_name":      self._current_patient.name,
            "patient_severity":  self._current_patient.true_severity,
            "chief_complaint":   COMPLAINTS[self._current_patient.observation.chief_complaint],
        }

        if self.render_mode == "human":
            self._render_reset(info)

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self._current_patient is not None, "Call reset() first"

        # Grade the decision
        grade = self._reward_calc.compute(action, self._current_patient)
        self._episode_log.append(grade)
        self._total_reward += grade.reward
        self._step_idx += 1

        # Termination check
        terminated = grade.critical_miss and self.cfg.get("time_pressure", False)
        truncated  = self._step_idx >= len(self._patient_queue)
        done       = terminated or truncated

        # Load next patient (if any)
        if not done and self._step_idx < len(self._patient_queue):
            self._current_patient = self._patient_queue[self._step_idx]
            obs = self._make_obs(self._current_patient)
        else:
            # Episode over — return last obs
            obs = self._make_obs(self._patient_queue[-1])

        info = {
            "action_name":      ACTION_NAMES[action],
            "correct":          grade.correct,
            "critical_miss":    grade.critical_miss,
            "over_triage":      grade.over_triage,
            "grade":            grade,
            "explanation":      grade.explanation,
            "total_reward":     round(self._total_reward, 2),
            "patients_done":    self._step_idx,
            "patients_total":   len(self._patient_queue),
        }

        # Add next patient info if not done
        if not done:
            info["next_patient"]   = self._current_patient.name
            info["next_severity"]  = self._current_patient.true_severity

        if self.render_mode == "human":
            self._render_step(action, grade)

        return obs, grade.reward, terminated, truncated, info

    def close(self):
        pass

    # ── Obs helper ────────────────────────────────────────────────────────────

    def _make_obs(self, patient: PatientRecord) -> np.ndarray:
        """Apply noise and return flat observation array."""
        obs = patient.observation

        # Add queue context for hard mode
        if self.cfg.get("queue", False):
            from dataclasses import replace
            obs = replace(obs, queue_length=len(self._patient_queue) - self._step_idx)

        noisy_obs = self._noise_model.apply(obs)
        return noisy_obs.to_array()

    # ── Config loader ─────────────────────────────────────────────────────────

    def _load_config(self, task_id: str, override: Optional[Dict] = None) -> Dict:
        """Load task config from YAML file, fall back to defaults."""
        tasks_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
        yaml_path = os.path.join(tasks_dir, f"{task_id}.yaml")

        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = dict(DEFAULT_CONFIGS.get(task_id, DEFAULT_CONFIGS["triage_easy"]))

        if override:
            cfg.update(override)
        return cfg

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode in ("human", "ansi") and self._current_patient:
            p   = self._current_patient
            obs = p.observation
            print(f"\n{'='*55}")
            print(f"  Patient: {p.name}  |  {p.age}{'F' if p.gender=='F' else 'M'}  |  Task: {self.task_id}")
            print(f"{'='*55}")
            print(f"  BP:    {obs.bp_systolic:.0f}/{obs.bp_diastolic:.0f} mmHg")
            print(f"  HR:    {obs.heart_rate:.0f} bpm")
            print(f"  SpO2:  {obs.spo2:.0f}%")
            print(f"  Temp:  {obs.temperature:.1f}°C")
            print(f"  RR:    {obs.resp_rate:.0f} breaths/min")
            print(f"  Complaint: {COMPLAINTS[obs.chief_complaint]}")
            print(f"  Pain:  {obs.pain_score}/10")
            print(f"  Queue: {obs.queue_length} waiting")
            print(f"{'='*55}")

    def _render_reset(self, info: Dict):
        print(f"\n{'='*55}")
        print(f"  NEW EPISODE | Task: {info['task_id']} | Patients: {info['patients_in_queue']}")
        print(f"{'='*55}")

    def _render_step(self, action: int, grade: GradeResult):
        status = "✅" if grade.correct else "❌"
        print(
            f"  {status} {ACTION_NAMES[action]:<12} | "
            f"Reward: {grade.reward:+.1f} | "
            f"{grade.patient_name} ({grade.severity})"
        )

    # ── Episode summary ───────────────────────────────────────────────────────

    def get_episode_report(self) -> EpisodeReport:
        """Build an EpisodeReport from current episode log."""
        log    = self._episode_log
        total  = len(log)
        if total == 0:
            return EpisodeReport(
                task_id=self.task_id, total_patients=0,
                correct_decisions=0, critical_misses=0, over_triages=0,
                total_reward=0.0, triage_accuracy=0.0,
                critical_miss_rate=0.0, over_triage_rate=0.0,
            )

        correct     = sum(1 for g in log if g.correct)
        crit_miss   = sum(1 for g in log if g.critical_miss)
        over_t      = sum(1 for g in log if g.over_triage)
        n_critical  = sum(1 for g in log if g.severity == SEVERITY_CRITICAL)
        n_stable    = sum(1 for g in log if g.severity == SEVERITY_STABLE)
        threshold   = self.cfg.get("success_threshold", 0.72)

        accuracy    = correct / total
        crit_rate   = crit_miss / max(1, n_critical)
        over_rate   = over_t   / max(1, n_stable)
        passed      = (accuracy >= threshold) and (crit_rate < 0.05)

        return EpisodeReport(
            task_id             = self.task_id,
            total_patients      = total,
            correct_decisions   = correct,
            critical_misses     = crit_miss,
            over_triages        = over_t,
            total_reward        = round(self._total_reward, 2),
            triage_accuracy     = round(accuracy, 3),
            critical_miss_rate  = round(crit_rate, 3),
            over_triage_rate    = round(over_rate, 3),
            decision_log        = log,
            passed              = passed,
        )
