"""
MedSense AI — Demo Script
Shows the environment running across all 3 tasks.

Run:
    python run_demo.py
    python run_demo.py triage_hard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medsense import (
    MedSenseEnv, TriageGrader,
    random_agent, rule_based_agent,
)


def single_episode_demo(task_id: str = "triage_easy"):
    print(f"\n{'='*55}")
    print(f"  MedSense AI — Single Episode Demo")
    print(f"  Task: {task_id}")
    print(f"{'='*55}")

    env = MedSenseEnv(task_id=task_id, render_mode="human")
    obs, info = env.reset(seed=42)

    print(f"\n  Patients this episode : {info['patients_in_queue']}")
    print(f"  First patient         : {info['patient_name']}")
    print(f"  Chief complaint       : {info['chief_complaint']}")
    env.render()

    terminated = truncated = False
    while not (terminated or truncated):
        action = rule_based_agent(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  → {info['action_name']:<12} | Reward: {reward:+.1f} | {info['explanation']}")

    report = env.get_episode_report()
    print(report.summary())
    env.close()


def benchmark_all_tasks():
    print(f"\n{'='*55}")
    print(f"  MedSense AI — Full Benchmark (50 eps per task)")
    print(f"{'='*55}")

    grader = TriageGrader(n_episodes=50, seed=42, verbose=False)

    results = grader.compare({
        "Random Agent":     random_agent,
        "Rule-Based Agent": rule_based_agent,
    })

    print(grader.leaderboard(results))


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "triage_easy"
    single_episode_demo(task)
    benchmark_all_tasks()
