import time
import os
import math
import random
import numpy as np
import torch
import socket

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import datetime

# 训练参数
EPISODES = 10000
MAX_STEPS = 2000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
GAMMA = 0.98
CLIP_EPSILON = 0.1
PPO_EPOCHS = 8
HIDDEN_SIZE = 256


class StarSectorEnv:
    """游戏环境 - 修改为连续动作空间"""

    def __init__(self, max_steps=1000):
        # Socket服务器初始化
        self.host = '127.0.0.1'
        self.port = 8888  # 改为8888端口以匹配Java端
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 设置socket选项，重用地址和增大缓冲区
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"PPO连续动作训练服务器已启动，监听 {self.host}:{self.port}")

        # 等待连接
        self.conn, self.addr = self.sock.accept()

        # 设置连接的缓冲区
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        print(f"来自 {self.addr} 的连接已建立")
        self.conn.setblocking(True)

        # 状态变量
        self.x, self.y, self.angle = 0, 0, 0
        self.vx, self.vy, self.vr = 0, 0, 0
        self.target_x, self.target_y, self.target_angle = 0, 0, 0
        self.rays = [0] * 16
        self.tactical_cooldown = 0
        self.tactical_available = 0
        self.tactical_active = 0

        # 训练状态
        self.episode_reward = 0
        self.steps = 0
        self.now_episode = 0
        self.max_steps = max_steps
        self.success = False
        self.next_actions = [0.0, 0.0, 0.0]  # [move, turn, strafe]

        self.in_target_zone = False  # 是否在目标区域内
        self.zone_steps = 0  # 在目标区域内停留的步数
        self.last_actions = None  # 记录上一个动作

    def reset(self):
        """重置环境"""
        # 发送重置指令（特殊动作值）
        self._send_actions([1.0, 0.0, 0.0])  # 暂时用特殊动作表示重置
        # 接收初始状态
        self._receive_full_state()
        # 重置训练状态
        self.episode_reward = 0
        self.steps = 0
        self.success = False

        self.previous_dist_to_target = None
        # 强制设置初始状态
        self.x = 0
        self.y = 1500
        self.angle = 0
        self.vx, self.vy, self.vr = 0, 0, 0
        self.last_x, self.last_y = self.x, self.y
        # 设置新目标
        self._reset_target()
        return self._get_state()

    def step(self, actions):
        """执行连续动作"""
        self.next_actions = actions

        # 接收新状态
        self._receive_full_state()

        # 计算奖励
        reward = self._get_basic_reward()
        self.episode_reward += reward
        self.steps += 1

        # 检查结束条件
        done = self.steps >= self.max_steps or self.success

        # 发送动作响应
        self._send_actions(self.next_actions)

        return self._get_state(), reward, done, {}

    def _send_actions(self, actions):
        """发送连续动作到游戏端"""
        try:
            # 格式: move:0.5,turn:-0.3,strafe:0.2
            move, turn, strafe = actions
            data = f"move:{move:.3f},turn:{turn:.3f},strafe:{strafe:.3f}\n"
            self.conn.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"发送动作错误: {e}")
            self.close()

    def _receive_full_state(self):
        """从游戏端接收完整状态"""
        try:
            data = b''
            while True:
                chunk = self.conn.recv(1024)
                if not chunk:
                    break
                data += chunk
                if b'\n' in data:
                    break

            if not data:
                raise ConnectionError("连接已关闭")

            decoded = data.decode('utf-8').strip()
            values = decoded.split(',')

            # 新的数据格式包含更多字段
            if len(values) < 25:  # 根据实际字段数调整
                print(f"警告: 数据字段不足，期望至少25个，得到{len(values)}个")
                # 继续处理，使用默认值填充缺失字段

            # 解析前验证数据
            for i in range(len(values)):
                if i >= len(values) or not values[i].strip():
                    values.append("0")

            # 解析状态数据
            try:
                # 基础位置和速度
                self.x = self._safe_float_parse(values[0])
                self.y = self._safe_float_parse(values[1])
                self.angle = self._safe_float_parse(values[2])
                self.vx = self._safe_float_parse(values[3])
                self.vy = self._safe_float_parse(values[4])
                self.vr = self._safe_float_parse(values[5])

                # 战术系统状态
                if len(values) > 8:
                    self.tactical_cooldown = self._safe_float_parse(values[6])
                    self.tactical_available = self._safe_float_parse(values[7])
                    self.tactical_active = self._safe_float_parse(values[8])

                # 激光数据
                if len(values) > 24:
                    for i in range(9, 25):
                        if i < len(values):
                            self.rays[i - 9] = self._safe_float_parse(values[i])

            except Exception as e:
                print(f"数据解析错误: {e}")
                self.x, self.y, self.angle, self.vx, self.vy, self.vr = 0, 0, 0, 0, 0, 0

        except Exception as e:
            print(f"接收状态错误: {e}")
            self.close()

    def _safe_float_parse(self, value):
        """安全的浮点数解析"""
        try:
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ValueError, TypeError):
            return 0.0

    def _reset_target(self):
        """重置目标位置"""
        if self.now_episode < 1000:
            min_distance = 300
            max_distance = 700
        elif self.now_episode < 3000:
            min_distance = 500
            max_distance = 1000
        else:
            min_distance = 700
            max_distance = 1300

        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_distance, max_distance)

        self.target_x = 0+distance * math.cos(angle)
        self.target_y = 1500+distance * math.sin(angle)
        self.target_angle = random.uniform(0, 360)

    def print_state_colored(self, state):
        """带颜色高亮的状态显示"""
        # ANSI颜色代码
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        END = '\033[0m'

        # 高亮位置和速度信息
        rel_x, rel_y, vx, vy, vr, *angles = state

        state_str = f"{RED}Pos:{rel_x:7.3f},{rel_y:7.3f} {GREEN}Vel:{vx:7.3f},{vy:7.3f} " \
                    f"{YELLOW}Rot:{vr:7.3f} {BLUE}Angles:{' '.join([f'{a:7.3f}' for a in angles])}{END}"

        print(f"\r{state_str}", end="", flush=True)

    def _get_state(self):
        """获取简化状态向量 - 保持11维"""
        try:
            # 验证基础数据
            if (math.isnan(self.x) or math.isnan(self.y) or
                    math.isnan(self.vx) or math.isnan(self.vy) or
                    math.isnan(self.vr)):
                self.x, self.y, self.angle, self.vx, self.vy, self.vr = 0, 0, 0, 0, 0, 0

            # 计算相对坐标
            dx = self.target_x - self.x
            dy = self.target_y - self.y

            # 限制距离范围
            max_safe_dist = 50000
            current_dist = math.sqrt(dx ** 2 + dy ** 2)
            if current_dist > max_safe_dist:
                self._reset_target()
                dx = self.target_x - self.x
                dy = self.target_y - self.y

            angle_rad = math.radians(self.angle)
            relative_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
            relative_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

            # 归一化
            max_dist = 20000
            relative_x_norm = max(-10.0, min(10.0, relative_x / (max_dist / 2)))
            relative_y_norm = max(-10.0, min(10.0, relative_y / (max_dist / 2)))

            # 角度使用sin/cos表示
            ship_angle_rad = math.radians(self.angle)
            target_angle_rad = math.radians(self.target_angle)

            # 计算目标相对于飞船的方向
            target_direction_rad = math.atan2(dy, dx)
            relative_direction_rad = target_direction_rad - ship_angle_rad

            # 归一化到[-π, π]
            while relative_direction_rad > math.pi:
                relative_direction_rad -= 2 * math.pi
            while relative_direction_rad < -math.pi:
                relative_direction_rad += 2 * math.pi

            # 构建简化状态向量 (11维)
            state = [
                # 相对位置和速度 (5维)
                relative_x_norm, relative_y_norm,
                max(-5.0, min(5.0, self.vx / 350)),  # MAX_SPEED = 350
                max(-5.0, min(5.0, self.vy / 350)),
                max(-5.0, min(5.0, self.vr / 20.0)),  # MAX_ANGULAR_VELOCITY = 20.0

                # 相对方向 (2维)
                math.sin(relative_direction_rad), math.cos(relative_direction_rad)
            ]

            self.print_state_colored(state)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            return np.zeros(7, dtype=np.float32)

    def _get_basic_reward(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        total_reward = 0

        # 1. 主要成功奖励 - 适应近距离
        if dist < 100:  # 减小成功距离要求
            base_reward = 200.0  # 提高基础奖励

            # 速度奖励 - 近距离需要更低速
            if speed <= 10:
                speed_bonus = 80.0
            elif speed <= 20:
                speed_bonus = 40.0
            else:
                speed_bonus = 10.0

            success_reward = base_reward + speed_bonus
            self.success = True
            self._reset_target()
            return success_reward

        # 2. 渐进式距离奖励 - 调整曲线适应近距离
        if dist < 1500:  # 减小距离范围
            distance_reward = 0.3 * (1.0 - (dist / 1500) ** 0.3)  # 更陡峭的曲线
            total_reward += distance_reward

        # 3. 距离变化奖励 - 提高敏感度
        if self.previous_dist_to_target is not None:
            dist_change = self.previous_dist_to_target - dist
            if dist_change > 1:  # 降低阈值
                approach_reward = 0.2 * min(1.0, dist_change / 30)  # 提高奖励
                total_reward += approach_reward
            elif dist_change < -10:
                distance_penalty = -0.1 * min(1.0, abs(dist_change) / 30)
                total_reward += distance_penalty

        self.previous_dist_to_target = dist

        # 4. 近距离速度控制 - 更严格
        if dist < 400:
            # 近距离强烈鼓励低速
            if speed < 15:
                low_speed_reward = 0.05 * (1 - speed / 15)
                total_reward += low_speed_reward
            elif speed > 25:
                high_speed_penalty = -0.03 * min(1.0, (speed - 25) / 20)
                total_reward += high_speed_penalty

        # 5. 动作平滑奖励
        if self.last_actions is not None:
            action_change = np.linalg.norm(np.array(self.last_actions) - np.array(self.next_actions))
            if action_change < 0.3:  # 更严格的平滑要求
                smooth_reward = 0.03
                total_reward += smooth_reward

        # 6. 时间惩罚 - 稍微降低，因为距离近了
        time_penalty = -0.003
        total_reward += time_penalty

        # 记录当前动作
        self.last_actions = self.next_actions.copy()

        return total_reward  # 放宽奖励范围限制

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        print("训练服务器已关闭")


# 连续动作PPO网络
class ContinuousPPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor - 输出动作的均值和标准差
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh()  # 限制在-1到1之间
        )

        self.actor_std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Softplus()  # 确保标准差为正
        )

        # Critic - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.shared(state)

        action_mean = self.actor_mean(features)
        action_std = self.actor_std(features) + 1e-6  # 避免除零
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)

        # 创建高斯分布
        dist = torch.distributions.Normal(action_mean, action_std)

        # 采样动作
        action = dist.sample()

        # 限制在-1到1之间
        action = torch.tanh(action)  # 使用tanh确保在-1到1之间

        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1)

        # 计算熵
        entropy = dist.entropy().sum(dim=-1)

        return action.squeeze().cpu().numpy(), log_prob, entropy, value.squeeze()


class ContinuousPPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64, log_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = ContinuousPPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.entropies = []

        # Tensorboard
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def get_action(self, state):
        """获取连续动作"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        return self.policy.get_action(state)

    def store_transition(self, state, action, log_prob, value, reward, done, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.entropies.append(entropy)

    def compute_advantages(self, next_value):
        if not self.rewards:
            return [], []

        advantages = []
        returns = []

        # 将数据转移到CPU进行numpy计算
        rewards = np.array(self.rewards)
        values = np.array([v.item() for v in self.values] + [next_value.item()])
        dones = np.array(self.dones)

        # 计算GAE (Generalized Advantage Estimation)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = advantages + values[:-1]

        # 归一化优势
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def update(self, next_value):
        if len(self.states) < self.batch_size:
            return 0, 0, 0

        advantages, returns = self.compute_advantages(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 计算新策略的动作概率
                action_means, action_stds, values = self.policy(batch_states)

                # 创建高斯分布
                dist = torch.distributions.Normal(action_means, action_stds)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # 概率比
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # 裁剪的PPO目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic损失 (MSE)
                critic_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()

                # 总损失
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # 更新
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy.item())

        # 清空缓冲区
        self.clear_buffer()

        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_entropy = np.mean(entropy_losses) if entropy_losses else 0

        return avg_actor_loss, avg_critic_loss, avg_entropy

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.entropies = []

    def save_model(self, path, episode=None, reward=None, success_rate=None):
        """保存模型，包含额外信息"""
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

        # 添加训练状态信息
        if episode is not None:
            save_dict['episode'] = episode
        if reward is not None:
            save_dict['reward'] = reward
        if success_rate is not None:
            save_dict['success_rate'] = success_rate

        torch.save(save_dict, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])

        # 如果优化器状态存在，加载它
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


def train_continuous_ppo():
    # 创建Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/ppo_continuous_{current_time}"
    writer = SummaryWriter(log_dir)

    # 创建环境
    env = StarSectorEnv(max_steps=MAX_STEPS)

    # 动态获取状态维度
    test_state = env.reset()
    state_dim = len(test_state)
    action_dim = 3  # 连续动作：move, turn, strafe

    print(f"检测到状态维度: {state_dim}")
    print(f"动作维度: {action_dim} (连续动作)")

    # 创建PPO智能体
    agent = ContinuousPPOAgent(state_dim, action_dim, lr=LEARNING_RATE, gamma=GAMMA,
                               clip_epsilon=CLIP_EPSILON, ppo_epochs=PPO_EPOCHS, batch_size=BATCH_SIZE)

    # 训练统计
    episode_rewards = []
    success_rates = []
    episode_lengths = []

    print("开始连续动作PPO训练...")

    try:
        for episode in range(EPISODES):
            state = env.reset()
            episode_reward = 0
            success = False

            for step in range(MAX_STEPS):
                # 选择连续动作
                actions, log_prob, entropy, value = agent.get_action(state)

                # 执行动作
                next_state, reward, done, _ = env.step(actions)

                # 存储经验
                agent.store_transition(state, actions, log_prob, value, reward, done, entropy)

                state = next_state
                episode_reward += reward

                if env.success:
                    success = True

                if done:
                    break

            # 回合结束，计算最终状态的价值
            with torch.no_grad():
                _, _, next_value = agent.policy(torch.FloatTensor(state).to(agent.device))

            # 更新策略
            actor_loss, critic_loss, entropy_loss = agent.update(next_value)

            # 记录统计
            episode_rewards.append(episode_reward)
            success_rates.append(1 if success else 0)
            episode_lengths.append(step + 1)

            # Tensorboard记录
            writer.add_scalar('Training/Episode Reward', episode_reward, episode)
            writer.add_scalar('Training/Success', 1 if success else 0, episode)
            writer.add_scalar('Training/Episode Length', step + 1, episode)

            if actor_loss > 0:
                writer.add_scalar('Loss/Actor', actor_loss, episode)
                writer.add_scalar('Loss/Critic', critic_loss, episode)
                writer.add_scalar('Loss/Entropy', entropy_loss, episode)

            # 输出训练信息
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
                success_rate = np.mean(success_rates[-10:]) * 100 if len(success_rates) >= 10 else (
                    100 if success else 0)
                avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else (step + 1)

                print(f"回合 {episode}/{EPISODES} | "
                      f"奖励: {episode_reward:.1f} (平均: {avg_reward:.1f}) | "
                      f"成功率: {success_rate:.1f}% | "
                      f"步骤: {step + 1} (平均: {avg_length:.1f})")

                # Tensorboard记录移动平均
                writer.add_scalar('Training/Average Reward (10 episodes)', avg_reward, episode)
                writer.add_scalar('Training/Success Rate (10 episodes)', success_rate, episode)
                writer.add_scalar('Training/Average Length (10 episodes)', avg_length, episode)

            # 保存模型
            if episode % 100 == 0 and episode > 0:
                if not os.path.exists("models"):
                    os.makedirs("models")
                save_path = f"models/ppo_continuous_model_episode_{episode}.pth"
                agent.save_model(save_path)
                print(f"模型已保存: {save_path}")

            env.now_episode += 1

    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        writer.close()

        # 保存最终模型
        agent.save_model("ppo_continuous_model_final.pth")
        print("连续动作训练完成，最终模型已保存")
        print(f"Tensorboard日志保存在: {log_dir}")
        print("使用命令查看: tensorboard --logdir=logs")


if __name__ == "__main__":
    train_continuous_ppo()