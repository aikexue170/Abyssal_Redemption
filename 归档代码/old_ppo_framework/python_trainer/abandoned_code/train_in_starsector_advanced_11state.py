import time
import os
import math
import random
import numpy as np
import torch
import socket
from torch.utils.tensorboard import SummaryWriter  # 新增导入
from abandoned_code.dqn_agent_advanced import AdvancedDQNAgent

# 训练参数
EPISODES = 2000
MAX_STEPS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 100
BUFFER_CAPACITY = 50000
MAX_SPEED = 350
MAX_ANGULAR_VELOCITY = 20.0


class StarSectorEnvWithTensorBoard:
    """带TensorBoard可视化的简化版游戏环境"""

    def __init__(self, max_steps=1000, experiment_name="dqn_experiment"):
        # Socket服务器初始化
        self.host = '127.0.0.1'
        self.port = 65432
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"DQN训练服务器已启动，监听 {self.host}:{self.port}")

        # 等待连接
        self.conn, self.addr = self.sock.accept()
        self.conn.setblocking(True)
        print(f"来自 {self.addr} 的连接已建立")

        # TensorBoard初始化
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(f"runs/{experiment_name}")
        print(f"TensorBoard日志目录: runs/{experiment_name}")

        # 状态变量
        self.x, self.y, self.angle = 0, 0, 0
        self.vx, self.vy, self.vr = 0, 0, 0
        self.target_x, self.target_y, self.target_angle = 0, 0, 0

        # 训练状态
        self.episode_reward = 0
        self.steps = 0
        self.now_episode = 0
        self.max_steps = max_steps
        self.success = False
        self.next_action = 0
        self.previous_dist_to_target = None

        # 训练监控
        self.q_value_history = []
        self.loss_history = []
        self.success_history = []
        self.gradient_history = []
        self.dead_neurons_history = []
        self.action_distribution = [0] * 7  # 7个动作的分布

        # TensorBoard记录频率
        self.log_interval = 10  # 每10个episode记录一次详细数据

    def reset(self):
        """重置环境"""
        self._send_action(100)
        self._receive_full_state()

        self.episode_reward = 0
        self.steps = 0
        self.success = False
        self.previous_dist_to_target = None

        # 强制设置初始状态
        self.x = 0
        self.y = 1500
        self.angle = 0
        self.vx, self.vy, self.vr = 0, 0, 0

        # 设置新目标
        self._reset_target()
        return self._get_state()

    def step(self, action):
        """执行动作"""
        self.next_action = action
        self._receive_full_state()

        # 记录动作分布
        self.action_distribution[action] += 1

        # 计算奖励
        reward = self._get_simplified_reward()
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

            # 只解析核心数据
            try:
                self.x = float(values[0])
                self.y = float(values[1])
                self.angle = float(values[2])
                self.vx = float(values[3])
                self.vy = float(values[4])
                self.vr = float(values[5])

            except Exception as e:
                print(f"数据解析错误: {e}")
                self.x, self.y, self.angle, self.vx, self.vy, self.vr = 0, 0, 0, 0, 0, 0

        except Exception as e:
            print(f"接收状态错误: {e}")
            self.close()

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

    def _get_state(self):
        """获取11维状态向量"""
        try:
            # 验证基础数据
            if any(math.isnan(val) for val in [self.x, self.y, self.angle, self.vx, self.vy, self.vr]):
                print("警告: 检测到NaN状态，使用默认值")
                self.x, self.y, self.angle, self.vx, self.vy, self.vr = 0, 0, 0, 0, 0, 0

            # 计算相对坐标
            dx = self.target_x - self.x
            dy = self.target_y - self.y

            # 限制距离范围，避免极端值
            max_safe_dist = 50000
            current_dist = math.sqrt(dx ** 2 + dy ** 2)
            if current_dist > max_safe_dist:
                print(f"警告: 距离过大 {current_dist:.1f}，重置目标")
                self._reset_target()
                dx = self.target_x - self.x
                dy = self.target_y - self.y

            angle_rad = math.radians(self.angle)
            relative_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
            relative_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

            # 归一化
            max_dist = 20000
            relative_x_norm = max(-5.0, min(5.0, relative_x / (max_dist / 2)))
            relative_y_norm = max(-5.0, min(5.0, relative_y / (max_dist / 2)))

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

            # 计算相对方向的sin/cos
            sin_relative_direction = math.sin(relative_direction_rad)
            cos_relative_direction = math.cos(relative_direction_rad)

            # 构建简化的状态向量 (11维)
            state = [
                relative_x_norm, relative_y_norm,
                max(-3.0, min(3.0, self.vx / MAX_SPEED)),
                max(-3.0, min(3.0, self.vy / MAX_SPEED)),
                max(-2.0, min(2.0, self.vr / MAX_ANGULAR_VELOCITY)),
                math.sin(ship_angle_rad), math.cos(ship_angle_rad),
                math.sin(target_angle_rad), math.cos(target_angle_rad),
                sin_relative_direction, cos_relative_direction
            ]

            # 最终状态验证
            for i, val in enumerate(state):
                if math.isnan(val) or math.isinf(val):
                    state[i] = 0.0

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            return np.zeros(11, dtype=np.float32)

    def _get_simplified_reward(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        # 计算角度差 - 放宽到±60度
        angle_diff = abs((self.angle - self.target_angle + 180) % 360 - 180)

        # 1. 到达目标奖励 - 宽松条件 + 非线性额外奖励
        if dist < 150 and angle_diff < 60:  # 放宽角度要求
            base_reward = 50.0  # 大幅奖励

            # 非线性速度奖励 - 鼓励低速接近
            if speed < 30:
                speed_bonus = 20 * (1 - speed / 10) ** 2  # 非线性：速度越低奖励越高
            elif speed < 50:
                speed_bonus = 10 * (1 - (speed - 10) / 15)  # 线性衰减
            else:
                speed_bonus = -5  # 高速接近惩罚

            # 非线性角度奖励 - 鼓励精确对齐
            if angle_diff < 30:
                angle_bonus = 15 * (1 - angle_diff / 10) ** 2  # 小角度时高奖励
            elif angle_diff < 60:
                angle_bonus = 8 * (1 - (angle_diff - 10) / 20)  # 中等角度中等奖励
            elif angle_diff < 90:
                angle_bonus = 3 * (1 - (angle_diff - 30) / 30)  # 大角度低奖励
            else:
                angle_bonus = 0

            # 时间奖励 - 鼓励快速完成
            time_bonus = 10 * (1 - self.steps / MAX_STEPS)

            arrival_reward = base_reward + speed_bonus + angle_bonus + time_bonus

            # 限制最大奖励但保持激励
            arrival_reward = min(arrival_reward, 60.0)  # 比之前高，提供更强激励

            self.success = True
            self._reset_target()
            return arrival_reward
        elif dist < 150:
            base_reward = 25

            # 非线性速度奖励 - 鼓励低速接近
            if speed < 30:
                speed_bonus = 20 * (1 - speed / 10) ** 2  # 非线性：速度越低奖励越高
            elif speed < 50:
                speed_bonus = 10 * (1 - (speed - 10) / 15)  # 线性衰减
            else:
                speed_bonus = -5  # 高速接近惩罚

            # 时间奖励 - 鼓励快速完成
            time_bonus = 10 * (1 - self.steps / MAX_STEPS)

            arrival_reward = base_reward + speed_bonus + time_bonus

            # 限制最大奖励但保持激励
            arrival_reward = min(arrival_reward, 60.0)  # 比之前高，提供更强激励

            self.success = True
            self._reset_target()
            return arrival_reward


        total_reward = 0

        # 2. 绝对距离奖励 - 提供持续的接近激励
        if dist < 4000:  # 4公里内都有奖励
            normalized_dist = min(1.0, dist / 4000)
            distance_reward = 0.25 * (1 - normalized_dist) ** 1.5  # 非线性，近距离奖励更多
            total_reward += distance_reward
        # 3. 距离变化奖励 - 更敏感的接近检测
        distance_change_reward = 0
        if self.previous_dist_to_target is not None:
            dist_change = self.previous_dist_to_target - dist

            if dist_change > 5:  # 降低阈值，更容易获得奖励
                # 非线性奖励：变化越大奖励越多
                distance_change_reward = 0.3 * min(1.0, (dist_change / 80) ** 0.7)
            elif dist_change < -30:  # 提高远离惩罚阈值
                distance_change_reward = -0.3 * min(1.0, abs(dist_change) / 100)

        self.previous_dist_to_target = dist
        total_reward += distance_change_reward
        # 4. 方向对齐奖励 - 更强的方向引导
        target_direction_rad = math.atan2(dy, dx)
        ship_angle_rad = math.radians(self.angle)
        angle_diff_rad = (target_direction_rad - ship_angle_rad) % (2 * math.pi)
        if angle_diff_rad > math.pi:
            angle_diff_rad -= 2 * math.pi

        alignment = math.cos(angle_diff_rad)

        # 非线性方向奖励 - 精确对齐奖励更多
        if alignment > 0.8:
            continuous_direction_reward = 0.12 * (alignment ** 2)  # 精确对齐时高奖励
        else:
            continuous_direction_reward = 0.06 * alignment  # 一般对齐中等奖励

        total_reward += continuous_direction_reward
        # 5. 移动效率奖励 - 鼓励在正确方向上移动
        movement_reward = 0
        if speed > 20:  # 降低速度阈值
            movement_direction_rad = math.atan2(self.vy, self.vx)
            movement_target_diff_rad = (target_direction_rad - movement_direction_rad) % (2 * math.pi)
            if movement_target_diff_rad > math.pi:
                movement_target_diff_rad -= 2 * math.pi

            movement_alignment = math.cos(movement_target_diff_rad)

            if movement_alignment > 0.5:  # 降低对齐阈值
                # 非线性移动奖励
                movement_reward = 0.2 * (movement_alignment ** 1.5)

        total_reward += movement_reward
        # 6. 智能行为奖励 - 新增组合奖励
        smart_behavior_reward = 0

        # 接近目标时的智能减速
        if dist < 1000:
            ideal_speed = max(10, min(80, dist / 12))  # 更宽松的理想速度
            speed_diff = abs(speed - ideal_speed)
            if speed_diff < 40:  # 放宽匹配范围
                speed_match_ratio = 1.0 - min(1.0, speed_diff / 30)
                smart_behavior_reward += 0.03 * (speed_match_ratio ** 2)

        # 旋转效率奖励
        if abs(angle_diff_rad) > 0.5:  # 约30度
            # 正确的旋转方向
            correct_rotation = (angle_diff_rad * self.vr) < 0
            if correct_rotation and abs(self.vr) > 2:  # 有意义的旋转
                rotation_efficiency = min(1.0, abs(self.vr) / 15)
                smart_behavior_reward += 0.02 * rotation_efficiency

        total_reward += smart_behavior_reward
        # 7. 探索奖励 - 鼓励尝试不同动作
        exploration_reward = 0
        if self.steps < 200:  # 早期训练阶段
            # 小幅鼓励动作变化
            if hasattr(self, 'last_action') and self.last_action != self.next_action:
                exploration_reward = 0.005

        self.last_action = self.next_action
        total_reward += exploration_reward
        # 8. 时间惩罚 - 轻微但存在
        time_penalty = -0.006
        total_reward += time_penalty
        # 合理的奖励范围 - 比之前宽松
        return max(-0.17, min(0.25, total_reward))

    def count_dead_neurons(self, agent, threshold=1e-6):
        """统计死亡神经元数量"""
        dead_count = 0
        total_neurons = 0

        for name, param in agent.policy_net.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                neuron_norms = torch.norm(param, dim=0)
                dead_neurons = (neuron_norms < threshold).sum().item()
                dead_count += dead_neurons
                total_neurons += param.shape[1]

        dead_ratio = dead_count / total_neurons if total_neurons > 0 else 0
        return dead_count, dead_ratio

    def log_to_tensorboard(self, agent, episode, total_reward, success, avg_loss):
        """记录数据到TensorBoard"""
        # 基本训练指标
        self.writer.add_scalar('Training/Episode_Reward', total_reward, episode)
        self.writer.add_scalar('Training/Success', 1 if success else 0, episode)
        self.writer.add_scalar('Training/Average_Loss', avg_loss, episode)
        self.writer.add_scalar('Training/Epsilon', agent.epsilon, episode)

        # 获取训练统计
        training_stats = agent.get_training_stats()
        if training_stats:
            self.writer.add_scalar('Training/Avg_Gradient_Norm', training_stats.get('avg_gradient_norm', 0), episode)
            self.writer.add_scalar('Training/Max_Gradient_Norm', training_stats.get('max_gradient_norm', 0), episode)
            self.writer.add_scalar('Training/Avg_Q_Range', training_stats.get('avg_q_range', 0), episode)
            self.writer.add_scalar('Training/Max_Q_Range', training_stats.get('max_q_range', 0), episode)

        # 死亡神经元统计
        dead_neurons, dead_ratio = self.count_dead_neurons(agent)
        self.writer.add_scalar('Network/Dead_Neurons', dead_neurons, episode)
        self.writer.add_scalar('Network/Dead_Neuron_Ratio', dead_ratio, episode)

        # 动作分布
        total_actions = sum(self.action_distribution)
        if total_actions > 0:
            action_probs = [count / total_actions for count in self.action_distribution]
            for action, prob in enumerate(action_probs):
                self.writer.add_scalar(f'Actions/Action_{action}', prob, episode)

            # 动作分布熵（衡量探索程度）
            action_entropy = -sum(p * math.log(p + 1e-8) for p in action_probs if p > 0)
            self.writer.add_scalar('Actions/Action_Entropy', action_entropy, episode)

        # 每100个episode记录一次网络权重直方图
        if episode % 100 == 0:
            for name, param in agent.policy_net.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param, episode)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, episode)

        # 每50个episode记录一次Q值分布
        if episode % 50 == 0:
            test_state = np.random.normal(0, 0.5, 11).astype(np.float32)
            with torch.no_grad():
                q_values = agent.policy_net(torch.FloatTensor(test_state).unsqueeze(0).to(agent.device))
                self.writer.add_histogram('Q_Values/Distribution', q_values, episode)

    def record_training_metrics(self, agent, loss, episode, total_reward, success):
        """记录训练指标并输出到TensorBoard"""
        # 记录到TensorBoard
        if episode % self.log_interval == 0:
            self.log_to_tensorboard(agent, episode, total_reward, success, loss)

        # 控制台输出（简化）
        if episode % 50 == 0:
            recent_success = np.mean(self.success_history[-50:]) if len(self.success_history) >= 50 else np.mean(
                self.success_history)
            print(f"Episode {episode} | Reward: {total_reward:.1f} | Success: {success} | "
                  f"Success Rate: {recent_success * 100:.1f}% | Epsilon: {agent.epsilon:.4f}")

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        self.writer.close()  # 关闭TensorBoard写入器
        print("训练服务器和TensorBoard已关闭")


# 主训练函数
def main():
    # 创建带TensorBoard的环境
    env = StarSectorEnvWithTensorBoard(max_steps=MAX_STEPS, experiment_name="dqn_advanced_v2")
    state_dim = 11
    action_dim = 7

    # 创建升级版智能体
    agent = AdvancedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE,
        buffer_capacity=BUFFER_CAPACITY,
        network_type="advanced"
    )

    # 训练统计
    episode_rewards = []
    episode_success = []

    print("开始带TensorBoard的DQN训练...")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"启动TensorBoard: tensorboard --logdir=runs")
    print(f"然后在浏览器中打开: http://localhost:6006")

    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            success = False
            episode_losses = []

            for step in range(MAX_STEPS):
                # 选择动作
                action = agent.select_action(state)

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                agent.push_memory(state, action, reward, next_state, done)

                # 训练并记录损失
                loss = agent.train()
                if loss > 0:
                    episode_losses.append(loss)

                # 更新状态
                state = next_state
                total_reward += reward

                if env.success:
                    success = True

                if done:
                    break

            # 更新探索率
            agent.update_epsilon()

            # 记录统计
            episode_rewards.append(total_reward)
            episode_success.append(1 if success else 0)
            env.now_episode = episode
            env.success_history.append(1 if success else 0)

            # 记录训练指标
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            env.record_training_metrics(agent, avg_loss, episode, total_reward, success)

            # 每200轮保存一次模型
            if episode % 200 == 0:
                if not os.path.exists("models_tensorboard"):
                    os.makedirs("models_tensorboard")
                save_path = f"models_tensorboard/dqn_tb_model_episode_{episode}.pth"
                agent.save(save_path, episode)
                print(f"模型已保存: {save_path}")

    except KeyboardInterrupt:
        print("训练被用户中断")
    finally:
        env.close()

        # 保存最终模型
        if not os.path.exists("models_tensorboard"):
            os.makedirs("models_tensorboard")
        agent.save("models_tensorboard/dqn_tb_final_model.pth")
        print("训练完成，最终模型已保存")


if __name__ == "__main__":
    main()
