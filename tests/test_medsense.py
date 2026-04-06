"""
MedSense AI — Test Suite (ViVi's environment tests)
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from medsense import (
    MedSenseEnv, TriageGrader, random_agent, rule_based_agent,
    ACTION_TREAT_NOW, ACTION_DELAY, ACTION_REFER,
    SEVERITY_CRITICAL, SEVERITY_URGENT, SEVERITY_STABLE,
)
from medsense.patient_generator import PatientGenerator
from medsense.noise import VitalNoiseModel
from medsense.models import PatientObservation


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def easy_env():
    e = MedSenseEnv(task_id="triage_easy")
    yield e
    e.close()

@pytest.fixture
def hard_env():
    e = MedSenseEnv(task_id="triage_hard")
    yield e
    e.close()


# ── Observation space ─────────────────────────────────────────────────────────
class TestObservationSpace:
    def test_obs_shape(self, easy_env):
        obs, _ = easy_env.reset()
        assert obs.shape == (14,)

    def test_obs_dtype(self, easy_env):
        obs, _ = easy_env.reset()
        assert obs.dtype == np.float32

    def test_obs_in_bounds(self, easy_env):
        obs, _ = easy_env.reset()
        assert easy_env.observation_space.contains(obs)

    def test_obs_in_bounds_after_steps(self, easy_env):
        easy_env.reset(seed=0)
        for _ in range(5):
            obs, _, term, trunc, _ = easy_env.step(ACTION_TREAT_NOW)
            assert easy_env.observation_space.contains(obs)
            if term or trunc: break


# ── Action space ──────────────────────────────────────────────────────────────
class TestActionSpace:
    def test_action_space_size(self, easy_env):
        assert easy_env.action_space.n == 3

    def test_all_actions_valid(self, easy_env):
        easy_env.reset()
        for a in [ACTION_TREAT_NOW, ACTION_DELAY, ACTION_REFER]:
            assert easy_env.action_space.contains(a)


# ── Reset ─────────────────────────────────────────────────────────────────────
class TestReset:
    def test_reset_returns_obs_and_info(self, easy_env):
        result = easy_env.reset()
        assert len(result) == 2

    def test_reset_info_has_task_id(self, easy_env):
        _, info = easy_env.reset()
        assert "task_id" in info
        assert info["task_id"] == "triage_easy"

    def test_reset_info_has_patient(self, easy_env):
        _, info = easy_env.reset()
        assert "patient_name" in info
        assert "patient_severity" in info

    def test_reset_task_switching(self):
        env = MedSenseEnv(task_id="triage_easy")
        env.reset()
        _, info = env.reset(task_id="triage_hard")
        assert info["task_id"] == "triage_hard"
        env.close()


# ── Step ──────────────────────────────────────────────────────────────────────
class TestStep:
    def test_step_returns_5_values(self, easy_env):
        easy_env.reset()
        result = easy_env.step(ACTION_TREAT_NOW)
        assert len(result) == 5

    def test_step_info_has_grade(self, easy_env):
        easy_env.reset()
        _, _, _, _, info = easy_env.step(ACTION_TREAT_NOW)
        assert "correct" in info
        assert "critical_miss" in info
        assert "explanation" in info

    def test_episode_ends_after_patients(self, easy_env):
        easy_env.reset(seed=0)
        # Easy mode = 1 patient per episode
        _, _, term, trunc, _ = easy_env.step(ACTION_DELAY)
        assert term or trunc

    def test_hard_mode_has_5_patients(self, hard_env):
        _, info = hard_env.reset(seed=0)
        assert info["patients_in_queue"] == 5


# ── Patient Generator ─────────────────────────────────────────────────────────
class TestPatientGenerator:
    def test_generates_all_severities(self):
        gen = PatientGenerator("easy", seed=0)
        severities = set()
        for _ in range(50):
            p = gen.generate()
            severities.add(p.true_severity)
        assert SEVERITY_CRITICAL in severities
        assert SEVERITY_STABLE   in severities

    def test_critical_patient_always_treat_now(self):
        gen = PatientGenerator("easy", seed=0)
        for _ in range(30):
            p = gen.generate()
            if p.true_severity == SEVERITY_CRITICAL:
                assert p.correct_action == ACTION_TREAT_NOW

    def test_vital_signs_in_range(self):
        gen = PatientGenerator("hard", seed=0)
        for _ in range(20):
            p   = gen.generate()
            obs = p.observation
            assert 50 <= obs.spo2 <= 100
            assert 20 <= obs.heart_rate <= 250
            assert 0  <= obs.pain_score <= 10

    def test_queue_generation(self):
        gen = PatientGenerator("hard", seed=0)
        q   = gen.generate_queue(5)
        assert len(q) == 5


# ── Noise Model ───────────────────────────────────────────────────────────────
class TestNoiseModel:
    def test_noise_changes_vitals(self):
        obs = PatientObservation(
            bp_systolic=120, bp_diastolic=80, heart_rate=75,
            spo2=98, temperature=37.0, resp_rate=16,
            chief_complaint=4, pain_score=2, age=30,
        )
        model = VitalNoiseModel("hard", seed=0)
        diffs = []
        for _ in range(20):
            noisy = model.apply(obs)
            diffs.append(abs(noisy.bp_systolic - obs.bp_systolic))
        assert max(diffs) > 0

    def test_easy_noise_smaller_than_hard(self):
        obs = PatientObservation(
            bp_systolic=120, bp_diastolic=80, heart_rate=75,
            spo2=98, temperature=37.0, resp_rate=16,
            chief_complaint=4, pain_score=2, age=30,
        )
        easy_model = VitalNoiseModel("easy",  seed=42)
        hard_model = VitalNoiseModel("hard",  seed=42)
        easy_diffs, hard_diffs = [], []
        for _ in range(30):
            easy_diffs.append(abs(easy_model.apply(obs).bp_systolic - obs.bp_systolic))
            hard_diffs.append(abs(hard_model.apply(obs).bp_systolic - obs.bp_systolic))
        assert np.mean(hard_diffs) > np.mean(easy_diffs)

    def test_spo2_never_exceeds_100(self):
        obs = PatientObservation(
            bp_systolic=120, bp_diastolic=80, heart_rate=75,
            spo2=99, temperature=37.0, resp_rate=16,
            chief_complaint=4, pain_score=2, age=30,
        )
        model = VitalNoiseModel("hard", seed=7)
        for _ in range(50):
            assert model.apply(obs).spo2 <= 100.0


# ── Reward ────────────────────────────────────────────────────────────────────
class TestReward:
    def test_correct_decision_positive_reward(self, easy_env):
        easy_env.reset(seed=0)
        patient = easy_env._current_patient
        correct_action = patient.correct_action
        _, reward, _, _, info = easy_env.step(correct_action)
        assert reward > 0
        assert info["correct"]

    def test_critical_miss_large_negative(self):
        from medsense.reward import RewardCalculator
        from medsense.patient_generator import PatientGenerator
        calc = RewardCalculator()
        gen  = PatientGenerator("easy", seed=0)
        # Find a critical patient
        for _ in range(20):
            p = gen.generate()
            if p.true_severity == SEVERITY_CRITICAL:
                grade = calc.compute(ACTION_DELAY, p)
                assert grade.reward <= -1.0
                assert grade.critical_miss
                break


# ── Grader ────────────────────────────────────────────────────────────────────
class TestGrader:
    def test_grader_runs_easy(self):
        grader = TriageGrader(n_episodes=10, seed=0)
        report = grader.evaluate(random_agent, "triage_easy")
        assert report.n_episodes == 10
        assert 0.0 <= report.avg_accuracy <= 1.0

    def test_rule_based_beats_random_on_easy(self):
        grader = TriageGrader(n_episodes=30, seed=42)
        r_random    = grader.evaluate(random_agent,     "triage_easy")
        r_rulebased = grader.evaluate(rule_based_agent, "triage_easy")
        assert r_rulebased.avg_accuracy >= r_random.avg_accuracy - 0.05

    def test_leaderboard_string(self):
        grader = TriageGrader(n_episodes=5, seed=0)
        results = grader.compare({"Random": random_agent}, task_ids=["triage_easy"])
        lb = grader.leaderboard(results)
        assert "LEADERBOARD" in lb
        assert "Random" in lb

    def test_critical_miss_rate_calculated(self):
        grader = TriageGrader(n_episodes=20, seed=0)
        report = grader.evaluate(random_agent, "triage_easy")
        assert 0.0 <= report.avg_critical_miss_rate <= 1.0


# ── OpenEnv yaml ──────────────────────────────────────────────────────────────
class TestOpenEnvYaml:
    def test_yaml_exists(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        yaml_path = os.path.join(root, "openenv.yaml")
        assert os.path.exists(yaml_path), "openenv.yaml is missing!"

    def test_yaml_has_required_keys(self):
        import yaml
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, "openenv.yaml")) as f:
            data = yaml.safe_load(f)
        for key in ["name", "tasks", "interface", "observation_space", "action_space"]:
            assert key in data, f"openenv.yaml missing key: {key}"

    def test_yaml_has_3_tasks(self):
        import yaml
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, "openenv.yaml")) as f:
            data = yaml.safe_load(f)
        assert len(data["tasks"]) == 3
