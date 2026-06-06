import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from dqn_model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=100000, batch_size=64, target_update=100,
                 checkpoint_path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0

        # 策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_capacity)

        # 检查点恢复
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"从检查点恢复: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        else:
            # 初始化目标网络
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            print("初始化新模型")

    def select_action(self, state, training=True):
        """使用ε-贪婪策略选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def push_memory(self, state, action, reward, next_state, done):
        """将经验存入回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """训练策略网络并返回损失"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换为PyTorch张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path, episode=None):
        """保存检查点"""
        checkpoint = {
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'episode': episode
        }

        torch.save(checkpoint, path)
        print(f"检查点已保存到 {path}")

    def save_lightweight(self, path):
        """轻量级保存，仅模型权重"""
        torch.save(self.policy_net.state_dict(), path)

    def load_checkpoint(self, path):
        """从检查点加载状态"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # 加载模型权重
            self.policy_net.load_state_dict(checkpoint['policy_state'])
            self.target_net.load_state_dict(checkpoint['target_state'])

            # 加载优化器状态
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # 加载训练状态
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']

            self.target_net.eval()

            print(f"模型恢复成功! ε={self.epsilon:.4f}, 更新计数={self.update_count}")
            return checkpoint.get('episode', 0)

        except Exception as e:
            print(f"加载检查点失败: {e}")
            # 尝试仅加载模型权重
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            print("仅模型权重加载成功")
            return 0