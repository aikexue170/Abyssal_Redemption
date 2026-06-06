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
EPISODES = 5000
MAX_STEPS = 1500
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4
HIDDEN_SIZE = 256


class StarSectorEnv:
    """游戏环境 - 修改为使用PPO"""

    def __init__(self, max_steps=1000):
        # Socket服务器初始化
        self.host = '127.0.0.1'
        self.port = 65432
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 设置socket选项，重用地址和增大缓冲区
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"PPO训练服务器已启动，监听 {self.host}:{self.port}")

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
        self.next_action = 0

        self.in_target_zone = False  # 是否在目标区域内
        self.zone_steps = 0  # 在目标区域内停留的步数
        self.last_action = None  # 记录上一个动作

    def reset(self):
        """重置环境"""
        # 发送重置指令
        self._send_action(100)
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

    def step(self, action):
        """执行动作"""
        self.next_action = action

        # 接收新状态
        self._receive_full_state()

        # 计算奖励
        reward = self._get_basic_reward()
        self.episode_reward += reward
        self.steps += 1

        # 检查结束条件
        done = self.steps >= self.max_steps or self.success

        # 发送动作响应
        self._send_action(self.next_action)

        return self._get_state(), reward, done, {}

    def _send_action(self, action):
        """发送动作到游戏端"""
        try:
            data = f"{action};{self.target_x};{self.target_y};{self.target_angle}\n"
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

            if len(values) < 25:
                raise ValueError(f"无效状态数据: {decoded}")

            # 解析前验证数据
            for i, value in enumerate(values):
                if not value.strip():
                    values[i] = "0"

            # 解析状态数据
            try:
                self.x = self._safe_float_parse(values[0])
                self.y = self._safe_float_parse(values[1])
                self.angle = self._safe_float_parse(values[2])
                self.vx = self._safe_float_parse(values[3])
                self.vy = self._safe_float_parse(values[4])
                self.vr = self._safe_float_parse(values[5])

                # 解析射线数据 - 暂时注释掉
                # self.rays = [self._safe_float_parse(values[i]) for i in range(6, 22)]

                # 解析战术系统状态 - 暂时注释掉
                # self.tactical_cooldown = int(float(values[22]) if values[22].strip() else 0)
                # self.tactical_available = int(float(values[23]) if values[23].strip() else 0)
                # self.tactical_active = int(float(values[24]) if values[24].strip() else 0)

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
        if self.now_episode < 500:
            min_distance = 1000
            max_distance = 1500
        elif self.now_episode < 1000:
            min_distance = 1500
            max_distance = 2000
        else:
            min_distance = 1500
            max_distance = 4000

        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_distance, max_distance)

        self.target_x = distance * math.cos(angle)
        self.target_y = distance * math.sin(angle)
        self.target_angle = random.uniform(0, 360)

    def print_state_colored(self, state):  # 添加state参数
        """
        带颜色高亮的状态显示
        """
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
        """获取简化状态向量 - 11维"""
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

                # 角度信息 (4维)
                # math.sin(ship_angle_rad), math.cos(ship_angle_rad),
                # math.sin(target_angle_rad), math.cos(target_angle_rad),

                # 相对方向 (2维)
                math.sin(relative_direction_rad), math.cos(relative_direction_rad)

                # 注释掉射线和战术系统状态
                # *[max(0.0, min(1.0, min(ray, 1) / 1)) for ray in self.rays],  # 16维射线
                # float(self.tactical_cooldown),  # 3维战术系统
                # float(self.tactical_available),
                # float(self.tactical_active)
            ]

            self.print_state_colored(state)

            # 状态验证
            for i, val in enumerate(state):
                if math.isnan(val) or math.isinf(val):
                    state[i] = 0.0

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            return np.zeros(11, dtype=np.float32)

    def _get_basic_reward(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        total_reward = 0

        # 1. 主要成功奖励 - 简化条件，专注到达
        if dist < 150:
            # 基础到达奖励（高额奖励）
            base_reward = 150.0

            # 速度奖励 - 鼓励适当速度到达，不是极低速
            # 15-25是最佳速度区间
            if 15 <= speed <= 25:
                speed_bonus = 50.0  # 最佳速度区间奖励
            elif 10 <= speed < 15 or 25 < speed <= 30:
                speed_bonus = 25.0  # 可接受速度区间奖励
            elif speed < 10:
                speed_bonus = 10.0  # 低速奖励较少
            else:
                speed_bonus = 5.0  # 高速奖励很少

            success_reward = base_reward + speed_bonus

            self.success = True
            self._reset_target()
            return success_reward

        # 2. 渐进式距离奖励 - 鼓励接近目标
        if dist < 3000:
            # 使用更陡峭的曲线，近距离奖励更高
            distance_reward = 0.2 * (1.0 - (dist / 3000) ** 0.5)
            total_reward += distance_reward

        # 3. 距离变化奖励 - 鼓励持续接近
        if self.previous_dist_to_target is not None:
            dist_change = self.previous_dist_to_target - dist

            if dist_change > 2:  # 靠近
                approach_reward = 0.15 * min(1.0, dist_change / 50)
                total_reward += approach_reward
            elif dist_change < -15:  # 明显远离
                distance_penalty = -0.08 * min(1.0, abs(dist_change) / 50)
                total_reward += distance_penalty

        self.previous_dist_to_target = dist

        # 4. 前进动作奖励 - 适度鼓励前进
        if self.last_action == 1:  # 前进动作
            forward_reward = 0.03
            total_reward += forward_reward

        # 5. 减速动作奖励 - 在适当情况下鼓励减速
        if self.last_action == 7:  # 减速动作
            # 在高速或接近目标时使用减速有奖励
            if speed > 40 or (dist < 500 and speed > 30):
                brake_reward = 0.04
                total_reward += brake_reward

        # 6. 速度控制奖励 - 只在近距离考虑
        if dist < 300:
            # 动态理想速度 - 避免过早减速
            if dist > 250:
                ideal_speed = 30  # 中近距离保持适当速度
            elif dist > 150:
                ideal_speed = 25  # 接近目标时开始减速
            else:
                ideal_speed = 20  # 非常接近时较低速度

            speed_diff = abs(speed - ideal_speed)
            if speed_diff < 15:
                speed_reward = 0.02 * (1 - speed_diff / 15)
                total_reward += speed_reward

        # 7. 时间惩罚 - 适度
        time_penalty = -0.005
        total_reward += time_penalty

        # 限制奖励范围
        return max(-0.1, min(0.5, total_reward))

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        print("训练服务器已关闭")


# PPO网络和智能体代码（从虚拟环境迁移）
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.state_dim = state_dim

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor - 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
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
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy, value.squeeze()


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64, log_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
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
        """获取动作 - 修复缺失的方法"""
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
        actions = torch.LongTensor(self.actions).to(self.device)
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
                logits, values = self.policy(batch_states)
                new_probs = torch.softmax(logits, dim=-1)
                new_dist = torch.distributions.Categorical(new_probs)
                new_log_probs = new_dist.log_prob(batch_actions)
                entropy = new_dist.entropy().mean()

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
            'state_dim': self.state_dim
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


def train_ppo_from_checkpoint(checkpoint_path, new_episodes=3000, lr=1e-4):
    """从检查点继续训练PPO"""

    # 创建环境
    env = StarSectorEnv()

    # 动态获取状态维度
    test_state = env.reset()
    state_dim = len(test_state)
    action_dim = 8

    print(f"检测到状态维度: {state_dim}")
    print(f"从检查点继续训练: {checkpoint_path}")

    # 创建Tensorboard日志目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/ppo_continued_{current_time}"

    # 创建PPO智能体
    agent = PPOAgent(state_dim, action_dim, lr=lr,clip_epsilon=0.1, log_dir=log_dir)

    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = agent.load_model(checkpoint_path)

        # 恢复训练状态
        start_episode = checkpoint.get('episode', 0) + 1
        print(f"从第 {start_episode} 回合继续训练")
    else:
        print(f"检查点不存在: {checkpoint_path}")
        start_episode = 0

    # 训练参数
    episodes = new_episodes
    max_steps = 1500
    save_interval = 100

    # 训练统计
    episode_rewards = []
    success_rates = []

    print("继续PPO训练...")

    try:
        for episode in range(start_episode, start_episode + episodes):
            state = env.reset()
            episode_reward = 0
            success = False

            for step in range(max_steps):
                # 选择动作 - 现在应该可以正常工作了
                action, log_prob, entropy, value = agent.get_action(state)

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                agent.store_transition(state, action, log_prob, value, reward, done, entropy)

                state = next_state
                episode_reward += reward

                if env.success:
                    success = True

                if done:
                    break

            # 回合结束，计算最终状态的价值
            with torch.no_grad():
                _, next_value = agent.policy(torch.FloatTensor(state).to(agent.device))

            # 更新策略
            actor_loss, critic_loss, entropy_loss = agent.update(next_value)

            # 记录统计
            episode_rewards.append(episode_reward)
            success_rates.append(1 if success else 0)

            # Tensorboard记录
            if agent.writer:
                agent.writer.add_scalar('Training/Episode Reward', episode_reward, episode)
                agent.writer.add_scalar('Training/Success', 1 if success else 0, episode)
                agent.writer.add_scalar('Training/Episode Length', step + 1, episode)

                if actor_loss > 0:
                    agent.writer.add_scalar('Loss/Actor', actor_loss, episode)
                    agent.writer.add_scalar('Loss/Critic', critic_loss, episode)
                    agent.writer.add_scalar('Loss/Entropy', entropy_loss, episode)

            # 输出训练信息
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
                success_rate = np.mean(success_rates[-10:]) * 100 if len(success_rates) >= 10 else (
                    100 if success else 0)

                print(f"回合 {episode}/{start_episode + episodes} | "
                      f"奖励: {episode_reward:.1f} (平均: {avg_reward:.1f}) | "
                      f"成功率: {success_rate:.1f}% | "
                      f"步骤: {step + 1}")

                # Tensorboard记录移动平均
                if agent.writer:
                    agent.writer.add_scalar('Training/Average Reward (10 episodes)', avg_reward, episode)
                    agent.writer.add_scalar('Training/Success Rate (10 episodes)', success_rate, episode)

            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                if not os.path.exists("models"):
                    os.makedirs("models")
                save_path = f"models/ppo_continued_model_episode_{episode}.pth"

                # 保存检查点，包含训练状态
                agent.save_model(save_path, episode=episode, reward=episode_reward, success_rate=success_rate)
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

        if agent.writer:
            agent.writer.close()

        # 保存最终模型
        final_save_path = "ppo_continued_model_final.pth"
        agent.save_model(final_save_path, episode=start_episode + episodes,
                         reward=episode_reward if 'episode_reward' in locals() else 0)
        print(f"继续训练完成，最终模型已保存: {final_save_path}")
        print(f"Tensorboard日志保存在: {log_dir}")


def train_ppo__for_new():
    # 创建Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/ppo_real_env_{current_time}"
    writer = SummaryWriter(log_dir)

    # 创建环境
    env = StarSectorEnv(max_steps=MAX_STEPS)

    # 动态获取状态维度
    test_state = env.reset()
    state_dim = len(test_state)
    action_dim = 8

    print(f"检测到状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")

    # 创建PPO智能体
    agent = PPOAgent(state_dim, action_dim, lr=LEARNING_RATE, gamma=GAMMA,
                     clip_epsilon=CLIP_EPSILON, ppo_epochs=PPO_EPOCHS, batch_size=BATCH_SIZE)

    # 训练统计
    episode_rewards = []
    success_rates = []
    episode_lengths = []

    print("开始PPO训练...")

    try:
        for episode in range(EPISODES):
            state = env.reset()
            episode_reward = 0
            success = False

            for step in range(MAX_STEPS):
                # 选择动作
                action, log_prob, entropy, value = agent.get_action(state)

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                agent.store_transition(state, action, log_prob, value, reward, done, entropy)

                state = next_state
                episode_reward += reward

                if env.success:
                    success = True

                if done:
                    break

            # 回合结束，计算最终状态的价值
            with torch.no_grad():
                _, next_value = agent.policy(torch.FloatTensor(state).to(agent.device))

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
                save_path = f"models/ppo_real_model_episode_{episode}.pth"
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
        agent.save_model("ppo_real_model_final.pth")
        print("训练完成，最终模型已保存")
        print(f"Tensorboard日志保存在: {log_dir}")
        print("使用命令查看: tensorboard --logdir=logs")


if __name__ == "__main__":
    #checkpoint_path = "previous_models/ppo_real_model_episode_400.pth"
    #train_ppo_from_checkpoint(checkpoint_path, new_episodes=10000, lr=1e-4)
    train_ppo__for_new()