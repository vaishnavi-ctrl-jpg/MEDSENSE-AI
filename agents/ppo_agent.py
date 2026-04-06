"""
MedSense AI — PPO Agent (ViVi's upgrade)
=========================================
Proximal Policy Optimisation — more stable than DQN for clinical tasks.
Uses Actor-Critic architecture with clipped surrogate objective.

WHY PPO OVER DQN:
  - Better sample efficiency in sparse reward environments
  - More stable training (clipped updates prevent catastrophic forgetting)
  - Actor-Critic gives both action probabilities AND value estimates
  - State-of-the-art for continuous and discrete action spaces

USAGE:
    python agents/ppo_agent.py triage_easy
    python agents/ppo_agent.py triage_hard --episodes 2000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from medsense.triage_env import MedSenseEnv
from medsense.grader import TriageGrader, rule_based_agent


# ── Actor-Critic Network ──────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared backbone with separate actor (policy) and critic (value) heads.
    Input:  14-dim patient observation
    Output: action probabilities + state value estimate
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        # Actor head — outputs action logits
        self.actor  = nn.Linear(64, action_dim)
        # Critic head — outputs scalar value estimate
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits   = self.actor(features)
        value    = self.critic(features)
        return logits, value

    def act(self, obs_tensor):
        logits, value = self.forward(obs_tensor)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs_batch, action_batch):
        logits, values = self.forward(obs_batch)
        dist    = Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        entropy   = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout of experience for PPO update."""

    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def push(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma=0.99, lam=0.95):
        """Generalised Advantage Estimation (GAE)."""
        returns, advantages = [], []
        gae = 0
        next_value = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae   = delta + gamma * lam * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            next_value = self.values[i]
        returns = [a + v for a, v in zip(advantages, self.values)]
        return (
            torch.FloatTensor(np.array(self.obs)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(returns),
            torch.FloatTensor(advantages),
            torch.stack(self.log_probs).detach(),
        )


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO with clipped objective, GAE advantage estimation,
    entropy bonus for exploration.
    """

    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        lr:          float = 3e-4,
        gamma:       float = 0.99,
        lam:         float = 0.95,
        clip_eps:    float = 0.2,
        n_epochs:    int   = 4,
        batch_size:  int   = 32,
        entropy_coef:float = 0.01,
        value_coef:  float = 0.5,
    ):
        self.name       = "PPO_Agent"
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.ent_coef   = entropy_coef
        self.val_coef   = value_coef

        self.net       = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

    def act(self, obs: np.ndarray, env=None) -> int:
        """Select action — compatible with TriageGrader.evaluate()."""
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            action, _, _ = self.net.act(obs_t)
        return action

    def act_with_info(self, obs: np.ndarray):
        """Select action and return log_prob + value for training."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action, log_prob, value = self.net.act(obs_t)
        return action, log_prob, value.item()

    def update(self):
        """Run PPO update on collected rollout."""
        obs_b, act_b, ret_b, adv_b, old_lp_b = self.buffer.compute_returns(
            self.gamma, self.lam
        )
        # Normalise advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        total_loss = 0
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs_b))
            for start in range(0, len(obs_b), self.batch_size):
                mb = idx[start:start + self.batch_size]
                log_probs, values, entropy = self.net.evaluate(obs_b[mb], act_b[mb])

                # Clipped surrogate loss
                ratio    = (log_probs - old_lp_b[mb]).exp()
                surr1    = ratio * adv_b[mb]
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b[mb]
                actor_l  = -torch.min(surr1, surr2).mean()

                # Value loss
                value_l  = ((values - ret_b[mb]) ** 2).mean()

                # Total loss
                loss = actor_l + self.val_coef * value_l - self.ent_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.clear()
        return total_loss

    def greedy_act(self, obs: np.ndarray, env=None) -> int:
        """Greedy (no sampling) — for evaluation."""
        with torch.no_grad():
            logits, _ = self.net(torch.FloatTensor(obs).unsqueeze(0))
        return int(torch.argmax(logits).item())


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_ppo(
    task_id:          str   = "triage_easy",
    episodes:         int   = 500,
    rollout_length:   int   = 64,
):
    print(f"\n{'='*55}")
    print(f"  MedSense PPO Training  |  task={task_id}")
    print(f"  Episodes={episodes}  |  rollout={rollout_length}")
    print(f"{'='*55}\n")

    env       = MedSenseEnv(task_id=task_id)
    state_dim = env.observation_space.shape[0]
    act_dim   = env.action_space.n
    agent     = PPOAgent(state_dim, act_dim)
    grader    = TriageGrader(n_episodes=50, seed=99)

    reward_log = []
    step_count = 0

    for ep in range(1, episodes + 1):
        obs, _    = env.reset(seed=ep)
        done      = False
        ep_reward = 0.0

        while not done:
            action, log_prob, value = agent.act_with_info(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(obs, action, r, log_prob, value, float(done))
            obs       = next_obs
            ep_reward += r
            step_count += 1

            # Update every rollout_length steps
            if step_count % rollout_length == 0:
                agent.update()

        reward_log.append(ep_reward)

        if ep % 100 == 0:
            report   = grader.evaluate(agent.greedy_act, task_id)
            avg_r    = sum(reward_log[-100:]) / 100
            print(
                f"  Ep {ep:>4}/{episodes} | "
                f"AvgReward={avg_r:>7.1f} | "
                f"WinRate={report.win_rate*100:>5.1f}% | "
                f"CritMiss={report.avg_critical_miss_rate*100:>4.1f}% | "
                f"{'✅PASS' if report.task_passed else '❌FAIL'}"
            )

    env.close()

    # Final comparison vs baselines
    print(f"\n{'='*55}")
    print("  Final Evaluation — PPO vs Baselines (100 episodes)")
    print(f"{'='*55}")
    final_grader = TriageGrader(n_episodes=100, seed=0)
    from agents.rule_based_agent import RuleBasedAgent
    rba = RuleBasedAgent()
    results = final_grader.compare({
        "PPO (trained)":    agent.greedy_act,
        "Rule-Based":       rule_based_agent,
    }, task_ids=[task_id])
    print(final_grader.leaderboard(results))
    return agent, reward_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="triage_easy",
                        choices=["triage_easy", "triage_medium", "triage_hard"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rollout", type=int, default=64)
    args = parser.parse_args()

    agent, log = train_ppo(args.task, args.episodes, args.rollout)
    save_path  = os.path.join(os.path.dirname(__file__), f"medsense_ppo_{args.task}.pth")
    torch.save(agent.net.state_dict(), save_path)
    print(f"\n  Model saved → {save_path}")
