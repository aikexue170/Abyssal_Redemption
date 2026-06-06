import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from dqn_model import create_model
from replay_buffer import create_buffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=100000, batch_size=64, target_update=100,
                 model_type="dqn", sequence_length=4, checkpoint_path=None):

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
        self.model_type = model_type
        self.sequence_length = sequence_length

        # 创建策略网络和目标网络
        self.policy_net = create_model(model_type, state_dim, action_dim).to(self.device)
        self.target_net = create_model(model_type, state_dim, action_dim).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 创建经验回放缓冲区
        if model_type == "dqn_lstm":
            self.memory = create_buffer("sequence", buffer_capacity, sequence_length=sequence_length)
        else:
            self.memory = create_buffer("standard", buffer_capacity)

        self.total_training_time = 0
        self.training_steps = 0

        # 初始化目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # LSTM相关状态
        self.current_hidden = None

        # 检查点恢复
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"从检查点恢复: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        else:
            print("初始化新模型")

    def select_action(self, state, training=True, tactical_cooldown=0):
        """使用ε-贪婪策略选择动作，添加战术系统冷却限制"""
        if training and np.random.random() < self.epsilon:
            # 随机探索时限制战术系统使用
            if tactical_cooldown > 0:
                # 如果战术系统在冷却，避免选择战术动作
                return np.random.randint(self.action_dim - 1)  # 排除战术动作
            else:
                # 即使不在冷却，也降低选择战术动作的概率
                if np.random.random() < 0.8:  # 80%的概率选择非战术动作
                    return np.random.randint(self.action_dim - 1)
                else:
                    return self.action_dim - 1  # 战术动作
        else:
            if self.model_type == "dqn_lstm":
                return self._select_action_lstm(state, tactical_cooldown)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)

                # 如果战术系统在冷却，给战术动作很大的负值
                if tactical_cooldown > 0:
                    q_values_clone = q_values.clone()
                    tactical_action_index = self.action_dim - 1  # 假设战术系统是最后一个动作
                    q_values_clone[0, tactical_action_index] = -1000
                    return q_values_clone.argmax().item()

                return q_values.argmax().item()

    def _select_action_lstm(self, state, tactical_cooldown=0):
        """LSTM版本的动作选择，添加战术冷却限制"""
        # 这里需要从memory中获取当前序列
        # 简化实现：假设外部已经维护了当前序列
        if hasattr(self.memory, 'current_sequence') and len(self.memory.current_sequence) >= self.sequence_length:
            # 获取当前序列
            sequence = list(self.memory.current_sequence)[-self.sequence_length:]
            sequence_states = [s for s, a, r, ns, d in sequence]

            # 构建输入序列
            sequence_tensor = torch.FloatTensor(np.array([sequence_states])).to(self.device)

            with torch.no_grad():
                q_values, self.current_hidden = self.policy_net(sequence_tensor, self.current_hidden)

            # 如果战术系统在冷却，屏蔽战术动作
            if tactical_cooldown > 0:
                q_values_clone = q_values.clone()
                tactical_action_index = self.action_dim - 1  # 假设战术系统是最后一个动作
                q_values_clone[0, tactical_action_index] = -1000
                return q_values_clone.argmax().item()

            return q_values.argmax().item()
        else:
            # 序列不足，随机选择（同样考虑冷却）
            if tactical_cooldown > 0:
                return np.random.randint(self.action_dim - 1)
            else:
                # 即使不在冷却，也降低选择战术动作的概率
                if np.random.random() < 0.8:
                    return np.random.randint(self.action_dim - 1)
                else:
                    return self.action_dim - 1

    def reset_hidden_state(self):
        """重置LSTM隐藏状态（在每个episode开始时调用）"""
        self.current_hidden = None

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def push_memory(self, state, action, reward, next_state, done):
        """将经验存入回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """训练策略网络并返回损失"""
        train_start = time.time()

        if self.model_type == "dqn_lstm":
            result = self._train_lstm()
        else:
            result = self._train_standard()

        train_time = time.time() - train_start

        return result

    def _train_standard(self):
        """标准DQN训练"""
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

    def _train_lstm(self):
        """LSTM-DQN训练"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # 从序列回放缓冲区采样
        sequences, actions, rewards, next_sequences, dones = self.memory.sample(self.batch_size)

        # 调试：打印所有输入的维度
        #print(f"DEBUG - sequences shape: {np.array(sequences).shape}")
        #print(f"DEBUG - actions shape: {np.array(actions).shape}")
        #print(f"DEBUG - rewards shape: {np.array(rewards).shape}")
        #print(f"DEBUG - next_sequences shape: {np.array(next_sequences).shape}")
        #print(f"DEBUG - dones shape: {np.array(dones).shape}")

        # 转换为PyTorch张量
        sequences = torch.FloatTensor(sequences).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_sequences = torch.FloatTensor(next_sequences).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 调试：打印转换后的维度
        #print(f"DEBUG - sequences tensor shape: {sequences.shape}")
        #print(f"DEBUG - actions tensor shape: {actions.shape}")
        #print(f"DEBUG - rewards tensor shape: {rewards.shape}")

        # 计算当前Q值
        current_q, _ = self.policy_net(sequences)
        #print(f"DEBUG - current_q shape: {current_q.shape}")

        # 修复维度问题
        #print(f"DEBUG - actions before unsqueeze: {actions.shape}")
        actions = actions.unsqueeze(1)
        #print(f"DEBUG - actions after unsqueeze: {actions.shape}")

        current_q = current_q.gather(1, actions).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q, _ = self.target_net(next_sequences)
            next_q = next_q.max(1)[0]  # 取最大Q值
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
            'episode': episode,
            'model_type': self.model_type
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