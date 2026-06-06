import torch
import torch.nn as nn
import torch.nn.functional as F

# 由于我也是人工智能新新新新手，所以会记相当多的笔记。应该是不影响阅读的。

class DQN(nn.Module):
    """
    DQN 神经网络模型
    输入: 状态向量
    输出: 每个动作的Q值
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    # 前向传播过程
    def forward(self, x):
        # 将x经过一次线性变换，也就是fc1层，然后将结果经过leaky_relu进行一次激活。大于0原样输出，小于0乘以0.1再输出。
        # leaky_relu相比传统relu，在负半轴区域保留了部分斜率，保证梯度不为0，能避免神经元死亡
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        return self.fc4(x)


class DQN_LSTM(nn.Module):
    """
    LSTM-based DQN 神经网络模型
    输入: 状态序列 (batch_size, sequence_length, state_dim)
    输出: 每个动作的Q值和新的隐藏状态
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=1):
        # num_layer是纵向堆叠的LSTM层的数量，如果为1，则代表只有一个LSTM层。一层已经足够捕捉一阶时序关系，加大了可能更拟合但是更难收敛
        super(DQN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        # 这里压缩一下，缩小到64维，减少参数量避免过拟合
        self.fc2 = nn.Linear(hidden_dim // 2, action_dim)

    # 前向传播过程
    def forward(self, x, hidden=None):
        # x形状: (batch_size, sequence_length, state_dim)
        batch_size = x.size(0)
        # 把batch维的数字取出来

        # 初始化隐藏状态。如果原先没有，就手动初始化一个全0的塞在同一个设备上
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)

        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)

        # 只取最后一个时间步的输出，喂给全连接层
        last_output = lstm_out[:, -1, :]
        x = F.leaky_relu(self.fc1(last_output), negative_slope=0.1)
        x = self.fc2(x)

        return x, hidden


# 简单的工厂函数，方便切换模型
def create_model(model_type, state_dim, action_dim, **kwargs):
    if model_type == "dqn":
        return DQN(state_dim, action_dim, **kwargs)
    elif model_type == "dqn_lstm":
        return DQN_LSTM(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")