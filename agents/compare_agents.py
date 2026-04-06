"""
MedSense AI — Multi-Algorithm Comparison (ViVi's upgrade)
==========================================================
Compares DQN, PPO, Rule-Based, and Random agents across all 3 tasks.
Produces a clean results table + saves reward curves as JSON for the dashboard.

USAGE:
    python agents/compare_agents.py              # all tasks, 100 episodes each
    python agents/compare_agents.py --task easy  # one task only
    python agents/compare_agents.py --episodes 50

OUTPUT:
    results/comparison_results.json   ← loaded by frontend dashboard
    results/comparison_table.txt      ← for README / submission
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import numpy as np

from medsense.triage_env import MedSenseEnv
from medsense.grader import TriageGrader, rule_based_agent, random_agent


def load_dqn(task_id: str):
    """Load saved DQN model if it exists, else return None."""
    try:
        import torch
        from agents.train import DQNAgent
        path = os.path.join(os.path.dirname(__file__), f"medsense_dqn_{task_id}.pth")
        if not os.path.exists(path):
            return None, None
        env   = MedSenseEnv(task_id=task_id)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        agent.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        agent.epsilon = 0.0
        env.close()
        return agent.greedy_act, "DQN (trained)"
    except Exception as e:
        return None, None


def load_ppo(task_id: str):
    """Load saved PPO model if it exists, else return None."""
    try:
        import torch
        from agents.ppo_agent import PPOAgent, ActorCritic
        path = os.path.join(os.path.dirname(__file__), f"medsense_ppo_{task_id}.pth")
        if not os.path.exists(path):
            return None, None
        env   = MedSenseEnv(task_id=task_id)
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
        agent.net.load_state_dict(torch.load(path, map_location="cpu"))
        env.close()
        return agent.greedy_act, "PPO (trained)"
    except Exception:
        return None, None


def run_comparison(task_ids: list, n_episodes: int = 100) -> dict:
    grader  = TriageGrader(n_episodes=n_episodes, seed=42)
    results = {}

    for task_id in task_ids:
        print(f"\n  Task: {task_id}")
        task_results = {}

        # Always available baselines
        policies = {
            "Random Agent":   random_agent,
            "Rule-Based Agent": rule_based_agent,
        }

        # Add trained models if available
        dqn_fn, dqn_name = load_dqn(task_id)
        ppo_fn, ppo_name = load_ppo(task_id)
        if dqn_fn: policies[dqn_name] = dqn_fn
        if ppo_fn: policies[ppo_name] = ppo_fn

        for name, policy in policies.items():
            print(f"    Evaluating: {name}")
            report = grader.evaluate(policy, task_id)
            task_results[name] = {
                "win_rate":        round(report.win_rate * 100, 1),
                "avg_accuracy":    round(report.avg_accuracy * 100, 1),
                "avg_crit_miss":   round(report.avg_critical_miss_rate * 100, 1),
                "avg_over_triage": round(report.avg_over_triage_rate * 100, 1),
                "avg_reward":      round(report.avg_reward, 2),
                "std_reward":      round(report.std_reward, 2),
                "task_passed":     report.task_passed,
                "reward_history":  [round(e.total_reward, 1) for e in report.episodes],
            }

        results[task_id] = task_results

    return results


def print_table(results: dict):
    bar = "=" * 80
    print(f"\n{bar}")
    print("  MEDSENSE AI — MULTI-ALGORITHM COMPARISON")
    print(bar)
    header = f"  {'Task':<16} {'Agent':<22} {'Acc%':>6} {'CritMiss%':>10} {'Reward':>9} {'Result':>8}"
    print(header)
    print("  " + "-" * 76)

    for task_id, agents in results.items():
        for agent_name, r in agents.items():
            status = "✅ PASS" if r["task_passed"] else "❌ FAIL"
            print(
                f"  {task_id:<16} {agent_name:<22} "
                f"{r['avg_accuracy']:>5.1f}% "
                f"{r['avg_crit_miss']:>9.1f}% "
                f"{r['avg_reward']:>9.2f} "
                f"{status:>8}"
            )
        print("  " + "-" * 76)

    print(bar)


def save_results(results: dict, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    # JSON for dashboard
    json_path = os.path.join(out_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # Text table for submission
    txt_path = os.path.join(out_dir, "comparison_table.txt")
    lines = []
    for task_id, agents in results.items():
        lines.append(f"\nTask: {task_id}")
        lines.append(f"{'Agent':<25} {'Accuracy':>10} {'CritMiss':>10} {'Reward':>10} {'Result':>8}")
        lines.append("-" * 65)
        for name, r in agents.items():
            status = "PASS" if r["task_passed"] else "FAIL"
            lines.append(
                f"{name:<25} {r['avg_accuracy']:>9.1f}% "
                f"{r['avg_crit_miss']:>9.1f}% "
                f"{r['avg_reward']:>10.2f} {status:>8}"
            )
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Table saved    → {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy","medium","hard","all"], default="all")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    task_map = {
        "easy":   ["triage_easy"],
        "medium": ["triage_medium"],
        "hard":   ["triage_hard"],
        "all":    ["triage_easy", "triage_medium", "triage_hard"],
    }
    task_ids = task_map[args.task]

    print(f"\n  MedSense AI — Algorithm Comparison")
    print(f"  Tasks: {task_ids} | Episodes: {args.episodes}")

    results = run_comparison(task_ids, args.episodes)
    print_table(results)
    save_results(results)
