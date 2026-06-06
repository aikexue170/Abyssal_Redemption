# dqn_model_advanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_Advanced(nn.Module):
    """
    升级版DQN网络 - 添加了层归一化和残差连接
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout_rate=0.1):
        super(DQN_Advanced, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # 层归一化

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, action_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化，保持梯度稳定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 第一层 + 层归一化 + 激活 + dropout
        residual = x
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.1)
        x = self.dropout(x)

        # 第二层 + 层归一化 + 激活 + dropout
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.1)
        x = self.dropout(x)

        # 第三层 + 层归一化 + 激活
        x = F.leaky_relu(self.ln3(self.fc3(x)), negative_slope=0.1)

        # 输出层
        x = self.fc4(x)

        return x


class DuelingDQN(nn.Module):
    """
    Dueling DQN网络 - 分离状态值和动作优势
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()

        # 共享的特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )

        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 动作优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 组合价值和建议：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
