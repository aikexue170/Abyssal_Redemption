import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """基础经验回放缓冲区，用于标准DQN"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


class SequenceReplayBuffer:
    """序列经验回放缓冲区，用于LSTM-DQN"""

    def __init__(self, capacity, sequence_length=4):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        self.current_sequence = deque(maxlen=sequence_length)

    def push(self, state, action, reward, next_state, done):
        # 将当前经验添加到序列
        self.current_sequence.append((state, action, reward, next_state, done))

        # 如果序列达到指定长度，保存到缓冲区
        if len(self.current_sequence) == self.sequence_length:
            sequence = list(self.current_sequence)
            self.buffer.append(sequence)

        # 如果episode结束，重置当前序列
        if done:
            self.current_sequence.clear()

    def sample(self, batch_size):
        # 采样批次序列
        batch_sequences = random.sample(self.buffer, batch_size)

        # 分离状态、动作等
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for sequence in batch_sequences:
            seq_states, seq_actions, seq_rewards, seq_next_states, seq_dones = zip(*sequence)

            # 确保我们取的是序列最后一个时间步的经验
            states.append(seq_states)  # 整个序列的状态
            actions.append(seq_actions[-1])  # 最后一个动作
            rewards.append(seq_rewards[-1])  # 最后一个奖励
            next_states.append(seq_next_states)  # 整个序列的下一个状态
            dones.append(seq_dones[-1])  # 最后一个done标志

        # 调试：检查返回的维度
        states_arr = np.array(states)
        actions_arr = np.array(actions)
        #print(f"BUFFER DEBUG - states array shape: {states_arr.shape}")
        #print(f"BUFFER DEBUG - actions array shape: {actions_arr.shape}")
        #print(f"BUFFER DEBUG - actions array sample: {actions_arr[:5]}")

        return (states_arr, actions_arr, np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# 工厂函数，方便切换缓冲区类型
def create_buffer(buffer_type, capacity, **kwargs):
    if buffer_type == "standard":
        return ReplayBuffer(capacity)
    elif buffer_type == "sequence":
        sequence_length = kwargs.get('sequence_length', 4)
        return SequenceReplayBuffer(capacity, sequence_length)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")