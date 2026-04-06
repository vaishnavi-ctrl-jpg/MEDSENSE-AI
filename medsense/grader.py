"""
MedSense AI — Task Grader
Hackathon-style grader with clinical validation logic.
Evaluates agents across all 3 tasks and produces pass/fail results.
"""

import numpy as np
from typing import Callable, Dict, Optional

from .triage_env import MedSenseEnv
from .models import (
    TaskReport, EpisodeReport,
    ACTION_TREAT_NOW, ACTION_DELAY, ACTION_REFER,
    SEVERITY_CRITICAL,
)


# ── Built-in baseline agents ──────────────────────────────────────────────────

def random_agent(obs: np.ndarray, env: MedSenseEnv) -> int:
    """Uniform random action."""
    return env.action_space.sample()


def rule_based_agent(obs: np.ndarray, env: MedSenseEnv) -> int:
    """
    Clinical rules baseline.
    Always treat_now for dangerous vital signs.
    """
    bp_sys    = obs[0]
    bp_dia    = obs[1]
    hr        = obs[2]
    spo2      = obs[3]
    temp      = obs[4]
    resp_rate = obs[5]
    complaint = int(obs[6])
    age       = obs[8]
    cardiac   = obs[9]

    # Hard clinical thresholds — always treat_now
    if spo2 < 92:
        return ACTION_TREAT_NOW
    if bp_sys > 180 or bp_sys < 85:
        return ACTION_TREAT_NOW
    if hr > 130 or hr < 45:
        return ACTION_TREAT_NOW
    if temp > 39.5:
        return ACTION_TREAT_NOW
    if resp_rate > 25:
        return ACTION_TREAT_NOW

    # Elevated + cardiac history → treat_now
    if cardiac == 1 and (hr > 105 or bp_sys > 155):
        return ACTION_TREAT_NOW

    # Specialist complaints → refer
    if complaint in {7, 8}:
        return ACTION_REFER

    # Borderline elevated → treat_now
    if bp_sys > 150 or hr > 105:
        return ACTION_TREAT_NOW

    # Otherwise → delay
    return ACTION_DELAY


# ── Grader ────────────────────────────────────────────────────────────────────

class TriageGrader:
    """
    Official hackathon-style grader for MedSense AI.

    Runs N episodes per task and computes:
      - Triage accuracy
      - Critical miss rate  ← KEY metric (must be < 5%)
      - Over-triage rate
      - Pass / Fail per task

    Usage:
        grader = TriageGrader(n_episodes=100)
        report = grader.evaluate(rule_based_agent, task_id="triage_hard")
        print(report.summary())
    """

    TASK_IDS = ["triage_easy", "triage_medium", "triage_hard"]

    def __init__(
        self,
        n_episodes: int = 100,
        seed:       Optional[int] = 42,
        verbose:    bool = False,
    ):
        self.n_episodes = n_episodes
        self.seed       = seed
        self.verbose    = verbose

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        policy:   Callable[[np.ndarray, MedSenseEnv], int],
        task_id:  str = "triage_easy",
        policy_name: str = "Agent",
    ) -> TaskReport:
        """Run N episodes of one task and return a TaskReport."""

        env     = MedSenseEnv(task_id=task_id)
        reports: list[EpisodeReport] = []

        for ep in range(self.n_episodes):
            ep_seed = (self.seed + ep) if self.seed is not None else None
            obs, _ = env.reset(seed=ep_seed)

            terminated = truncated = False
            while not (terminated or truncated):
                action = policy(obs, env)
                obs, _, terminated, truncated, _ = env.step(action)

            report = env.get_episode_report()
            reports.append(report)

            if self.verbose:
                status = "PASS ✅" if report.passed else "FAIL ❌"
                print(
                    f"  Ep {ep+1:>4}/{self.n_episodes} | {status} | "
                    f"Acc: {report.triage_accuracy*100:.0f}% | "
                    f"CritMiss: {report.critical_miss_rate*100:.0f}% | "
                    f"Reward: {report.total_reward:>7.2f}"
                )

        env.close()
        return self._aggregate(task_id, reports)

    # ── Aggregate ─────────────────────────────────────────────────────────────

    def _aggregate(self, task_id: str, reports: list) -> TaskReport:
        n = len(reports)
        rewards    = [r.total_reward        for r in reports]
        accuracies = [r.triage_accuracy     for r in reports]
        crit_rates = [r.critical_miss_rate  for r in reports]
        over_rates = [r.over_triage_rate    for r in reports]
        wins       = [r.passed              for r in reports]

        avg_accuracy  = float(np.mean(accuracies))
        avg_crit_miss = float(np.mean(crit_rates))
        task_passed   = (
            float(np.mean(wins)) >= 0.5 and
            avg_crit_miss < 0.05
        )

        return TaskReport(
            task_id                 = task_id,
            n_episodes              = n,
            win_rate                = float(np.mean(wins)),
            avg_reward              = float(np.mean(rewards)),
            avg_accuracy            = avg_accuracy,
            avg_critical_miss_rate  = avg_crit_miss,
            avg_over_triage_rate    = float(np.mean(over_rates)),
            std_reward              = float(np.std(rewards)),
            task_passed             = task_passed,
            episodes                = reports,
        )

    # ── Multi-task comparison ─────────────────────────────────────────────────

    def compare(
        self,
        policies: Dict[str, Callable],
        task_ids: Optional[list] = None,
    ) -> Dict[str, Dict[str, TaskReport]]:
        """
        Evaluate multiple agents across multiple tasks.
        Returns: { policy_name: { task_id: TaskReport } }
        """
        task_ids = task_ids or self.TASK_IDS
        results  = {}
        for name, policy in policies.items():
            print(f"\n  Evaluating: {name}")
            results[name] = {}
            for task_id in task_ids:
                print(f"    Task: {task_id}")
                results[name][task_id] = self.evaluate(policy, task_id, name)
        return results

    def leaderboard(
        self,
        results: Dict[str, Dict[str, TaskReport]],
    ) -> str:
        """Print a leaderboard across all agents and tasks."""
        bar = "=" * 75
        lines = [bar, "  MEDSENSE AI — LEADERBOARD", bar]
        header = f"  {'Agent':<18} {'Task':<16} {'Acc%':>6} {'CritMiss%':>10} {'Reward':>9} {'Result':>8}"
        lines += [header, "  " + "-" * 71]

        for agent_name, task_reports in results.items():
            for task_id, report in task_reports.items():
                status = "✅ PASS" if report.task_passed else "❌ FAIL"
                row = (
                    f"  {agent_name:<18} {task_id:<16} "
                    f"{report.avg_accuracy*100:>5.1f}% "
                    f"{report.avg_critical_miss_rate*100:>9.1f}% "
                    f"{report.avg_reward:>9.2f} "
                    f"{status:>8}"
                )
                lines.append(row)
            lines.append("  " + "-" * 71)

        lines.append(bar)
        return "\n".join(lines)
