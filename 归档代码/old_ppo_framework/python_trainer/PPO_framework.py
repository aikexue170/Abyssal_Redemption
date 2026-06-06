import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor层 - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )

        # Critic层 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages(self, next_value):
        advantages = []
        returns = []

        # 将数据转移到CPU进行numpy计算
        rewards = np.array(self.rewards)
        values = np.array([v.item() for v in self.values] + [next_value.item()])
        dones = np.array(self.dones)

        # 计算GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # 0.95是GAE参数
            advantages.insert(0, gae)

        returns = advantages + values[:-1]

        # 归一化优势
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def update(self, next_value):
        advantages, returns = self.compute_advantages(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # 多轮PPO更新
        for _ in range(4):  # PPO epoch数
            # 计算新策略的动作概率
            logits, values = self.policy(states)
            new_probs = torch.softmax(logits, dim=-1)
            new_dist = torch.distributions.Categorical(new_probs)
            new_log_probs = new_dist.log_prob(actions)
            entropy = new_dist.entropy().mean()

            # 概率比
            ratio = (new_log_probs - old_log_probs).exp()

            # 裁剪的PPO目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic损失 (MSE)
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()

            # 总损失
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
            self.optimizer.step()

        # 清空缓冲区
        self.clear_buffer()

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []