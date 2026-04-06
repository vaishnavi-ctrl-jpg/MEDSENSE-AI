"""
MedSense AI — DQN Training v2.0  (Nidz upgraded)
=================================================
Changes in v2:
  - Experience replay buffer (was missing — made DQN unstable)
  - Periodic clinical metric evaluation every 100 episodes
  - Final comparison vs Rule-Based agent
  - Saves reward curve to results/reward_curve_{task}.json
  - Proper argparse with choices validation

USAGE:
    python agents/train.py triage_easy
    python agents/train.py triage_medium --episodes 1000
    python agents/train.py triage_hard   --episodes 2000
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, random, json
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from medsense.triage_env import MedSenseEnv
from medsense.grader import TriageGrader, rule_based_agent


# ── Network ───────────────────────────────────────────────────────────────────
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    def forward(self, x): return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)
    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (torch.FloatTensor(np.array(obs)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_obs)),
                torch.FloatTensor(dones))
    def __len__(self): return len(self.buf)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64):
        self.name       = "DQN_Agent"
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size
        self.epsilon    = 1.0
        self.eps_end    = 0.05
        self.eps_decay  = 0.995
        self.q_net      = DQNetwork(state_dim, action_dim)
        self.target     = DQNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.q_net.state_dict())
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()
        self.buffer     = ReplayBuffer()
        self._steps     = 0

    def act(self, obs, env=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(obs).unsqueeze(0))
        return int(torch.argmax(q).item())

    def greedy_act(self, obs, env=None):
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(obs).unsqueeze(0))
        return int(torch.argmax(q).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size: return None
        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)
        q_vals   = self.q_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_obs).max(1)[0]
            target = rews + self.gamma * next_q * (1 - dones)
        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self._steps += 1
        if self._steps % 200 == 0:
            self.target.load_state_dict(self.q_net.state_dict())
        return loss.item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train(task_id="triage_easy", episodes=500):
    print(f"\n{'='*55}")
    print(f"  MedSense DQN v2  |  task={task_id}  |  eps={episodes}")
    print(f"{'='*55}\n")

    env       = MedSenseEnv(task_id=task_id)
    agent     = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    grader    = TriageGrader(n_episodes=50, seed=99)
    reward_log = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        done   = False; ep_r = 0.0
        while not done:
            action = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(obs, action, r, next_obs, float(done))
            agent.train_step()
            obs = next_obs; ep_r += r
        reward_log.append(ep_r)

        if ep % 100 == 0:
            report = grader.evaluate(agent.greedy_act, task_id)
            avg_r  = sum(reward_log[-100:]) / 100
            print(f"  Ep {ep:>4}/{episodes} | ε={agent.epsilon:.3f} | "
                  f"AvgR={avg_r:>7.1f} | "
                  f"WinRate={report.win_rate*100:>5.1f}% | "
                  f"CritMiss={report.avg_critical_miss_rate*100:>4.1f}% | "
                  f"{'✅PASS' if report.task_passed else '❌FAIL'}")

    env.close()

    # Save reward curve
    os.makedirs("results", exist_ok=True)
    curve_path = f"results/reward_curve_dqn_{task_id}.json"
    with open(curve_path, "w") as f:
        json.dump({"task_id": task_id, "algorithm": "DQN",
                   "episodes": list(range(1, len(reward_log)+1)),
                   "rewards": [round(r,2) for r in reward_log]}, f)
    print(f"\n  Reward curve → {curve_path}")

    # Final comparison
    print(f"\n{'='*55}")
    print("  Final Evaluation — DQN vs Rule-Based (100 episodes)")
    print(f"{'='*55}")
    final_g = TriageGrader(n_episodes=100, seed=0)
    for name, policy in [("DQN (greedy)", agent.greedy_act), ("Rule-Based", rule_based_agent)]:
        r = final_g.evaluate(policy, task_id)
        status = "✅ PASS" if r.task_passed else "❌ FAIL"
        print(f"  {name:<18} acc={r.avg_accuracy*100:.0f}% "
              f"crit={r.avg_critical_miss_rate*100:.0f}% "
              f"reward={r.avg_reward:.2f} {status}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="triage_easy",
                        choices=["triage_easy","triage_medium","triage_hard"])
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    agent = train(args.task, args.episodes)
    save_path = os.path.join(os.path.dirname(__file__), f"medsense_dqn_{args.task}.pth")
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"  Model saved → {save_path}")