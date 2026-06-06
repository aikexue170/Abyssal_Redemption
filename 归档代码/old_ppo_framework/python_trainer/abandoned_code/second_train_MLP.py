import time
import os
import math
import random
import numpy as np
import torch
import socket
from abandoned_code.dqn_agent_MLP import DQNAgent

# 二次训练参数 - 调整以优化训练
EPISODES = 4000  # 彻夜！启动!!!
MAX_STEPS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 4e-5  # 降低学习率进行微调
GAMMA = 0.99
EPSILON_START = 0.06  # 从较低探索率开始
EPSILON_END = 0.005
EPSILON_DECAY = 0.997  # 更慢的探索率衰减
TARGET_UPDATE = 300
BUFFER_CAPACITY = 200000
MAX_SPEED = 350
MAX_ANGULAR_VELOCITY = 20.0
RAY_MAX_DISTANCE = 1


# 舰船重置的坐标
RESET_LOCATION_X = 0
RESET_LOCATION_Y = 1500

# 模型加载路径
LOAD_MODEL_PATH = "models_second/dqn_second_final_model.pth"  # 根据实际情况修改


class StarSectorEnv:
    """二次训练环境 - 使用优化后的奖励函数"""

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
        print(f"二次训练服务器已启动，监听 {self.host}:{self.port}")

        # 等待连接
        self.conn, self.addr = self.sock.accept()
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

        # 新增：用于轨迹效率计算
        self.previous_dist_to_target = None
        self.previous_angle_diff = None

    def reset(self):
        """重置环境"""
        self._send_action(100)
        self._receive_full_state()

        self.episode_reward = 0
        self.steps = 0
        self.success = False
        self.previous_dist_to_target = None
        self.previous_angle_diff = None

        # 设置新目标 - 使用挑战性目标生成
        self._reset_target_challenging()
        return self._get_state()

    def step(self, action):
        """执行动作"""
        self.next_action = action
        self._receive_full_state()

        # 使用优化后的奖励函数
        reward = self._get_nonlinear_reward()
        self.episode_reward += reward
        self.steps += 1

        done = self.steps >= self.max_steps or self.success
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

            # 解析状态数据
            for i, value in enumerate(values):
                if not value.strip():
                    values[i] = "0"

            try:
                self.x = self._safe_float_parse(values[0])
                self.y = self._safe_float_parse(values[1])
                self.angle = self._safe_float_parse(values[2])
                self.vx = self._safe_float_parse(values[3])
                self.vy = self._safe_float_parse(values[4])
                self.vr = self._safe_float_parse(values[5])
                self.rays = [self._safe_float_parse(values[i]) for i in range(6, 22)]
                self.tactical_cooldown = int(float(values[22]) if values[22].strip() else 0)
                self.tactical_available = int(float(values[23]) if values[23].strip() else 0)
                self.tactical_active = int(float(values[24]) if values[24].strip() else 0)

            except Exception as e:
                print(f"数据解析错误: {e}")
                self.x, self.y, self.angle, self.vx, self.vy, self.vr = 0, 0, 0, 0, 0, 0
                self.rays = [0] * 16
                self.tactical_cooldown, self.tactical_available, self.tactical_active = 0, 0, 0

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

    def _reset_target_challenging(self):
        """挑战性目标生成 - 促进精细控制"""
        # 60%近距离精细控制，40%正常距离
        if random.random() < 0.6:
            min_distance = 200
            max_distance = 800
        else:
            min_distance = 1000
            max_distance = 2500

        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_distance, max_distance)

        self.target_x = RESET_LOCATION_X + distance * math.cos(angle)
        self.target_y = RESET_LOCATION_Y + distance * math.sin(angle)
        self.target_angle = random.uniform(0, 360)

    def _get_state(self):
        """获取30维状态向量"""
        try:
            # 计算相对坐标
            dx = self.target_x - self.x
            dy = self.target_y - self.y

            # 限制距离范围
            max_safe_dist = 50000
            current_dist = math.sqrt(dx ** 2 + dy ** 2)
            if current_dist > max_safe_dist:
                self._reset_target_challenging()
                dx = self.target_x - self.x
                dy = self.target_y - self.y

            angle_rad = math.radians(self.angle)
            relative_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
            relative_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

            # 归一化
            max_dist = 10000
            relative_x_norm = max(-10.0, min(10.0, relative_x / (max_dist / 2)))
            relative_y_norm = max(-10.0, min(10.0, relative_y / (max_dist / 2)))

            # 角度计算
            ship_angle_rad = math.radians(self.angle)
            target_angle_rad = math.radians(self.target_angle)

            # 计算相对方向
            target_direction_rad = math.atan2(dy, dx)
            relative_direction_rad = target_direction_rad - ship_angle_rad
            while relative_direction_rad > math.pi:
                relative_direction_rad -= 2 * math.pi
            while relative_direction_rad < -math.pi:
                relative_direction_rad += 2 * math.pi

            sin_relative_direction = math.sin(relative_direction_rad)
            cos_relative_direction = math.cos(relative_direction_rad)

            # 构建状态向量
            state = [
                # 相对位置和速度 (9维)
                relative_x_norm, relative_y_norm,
                max(-5.0, min(5.0, self.vx / MAX_SPEED)),
                max(-5.0, min(5.0, self.vy / MAX_SPEED)),
                max(-5.0, min(5.0, self.vr / MAX_ANGULAR_VELOCITY)),
                math.sin(ship_angle_rad), math.cos(ship_angle_rad),
                math.sin(target_angle_rad), math.cos(target_angle_rad),

                sin_relative_direction, cos_relative_direction,

                # 射线距离 (16维)
                *[max(0.0, min(1.0, min(ray, RAY_MAX_DISTANCE) / RAY_MAX_DISTANCE)) for ray in self.rays],

                # 战术系统状态 (3维)
                float(self.tactical_cooldown),
                float(self.tactical_available),
                float(self.tactical_active)
            ]

            # 状态验证
            for i, val in enumerate(state):
                if math.isnan(val) or math.isinf(val):
                    state[i] = 0.0

            # 输出状态预览
            actual_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            relative_direction_deg = math.degrees(relative_direction_rad)

            preview_str = f"距离: {current_dist:6.1f} | 速度: {actual_speed:5.1f} | "
            preview_str += f"相对位置: ({relative_x_norm:5.2f}, {relative_y_norm:5.2f}) | "
            preview_str += f"相对方向: {relative_direction_deg:6.1f}°"

            print(f"\r{preview_str}", end="", flush=True)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            return np.zeros(30, dtype=np.float32)

    def _get_nonlinear_reward(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        # 修复角度差计算
        angle_diff = abs((self.angle - self.target_angle + 180) % 360 - 180)
        # 1. 到达目标奖励 - 放宽条件，提高奖励
        if dist < 120 and angle_diff < 15:  # 放宽条件：120米内且15度内
            base_reward = 50.0  # 大幅提高基础奖励
            # 非线性速度奖励
            if speed < 10:
                speed_bonus = 30
            elif speed < 20:
                speed_bonus = 20
            elif speed < 30:
                speed_bonus = 10
            elif speed < 50:
                speed_bonus = 5
            else:
                speed_bonus = -5
            # 非线性角度奖励
            if angle_diff < 5:
                angle_bonus = 20
            elif angle_diff < 10:
                angle_bonus = 15
            elif angle_diff < 15:
                angle_bonus = 5
            time_bonus = 15 * (1 - self.steps / MAX_STEPS)
            arrival_reward = base_reward + speed_bonus + angle_bonus + time_bonus
            self.success = True
            self._reset_target_challenging()
            return arrival_reward
        total_reward = 0
        # 2. 大幅降低稀疏奖励的系数
        # 距离奖励
        if dist < 5000:
            normalized_dist = min(1.0, dist / 5000)
            distance_reward = 0.08 * (1 - normalized_dist) ** 2  # 从0.15降到0.08
            total_reward += distance_reward
        # 3. 距离变化奖励 - 降低
        distance_change_reward = 0
        if self.previous_dist_to_target is not None:
            dist_change = self.previous_dist_to_target - dist

            if dist_change > 0:
                distance_change_reward = 0.02 * min(1.0, dist_change / 100)  # 从0.05降到0.02
            elif dist_change < -20:
                distance_change_reward = -0.01 * min(1.0, abs(dist_change) / 100)  # 从0.025降到0.01

        self.previous_dist_to_target = dist
        total_reward += distance_change_reward
        # 4. 速度奖励 - 降低
        speed_reward = 0

        if dist < 300:
            ideal_speed_min, ideal_speed_max = 10, 50
            speed_importance = 0.8  # 降低重要性
        elif dist < 1000:
            ideal_speed_min, ideal_speed_max = 30, 70
            speed_importance = 0.5
        else:
            ideal_speed_min, ideal_speed_max = 50, 100
            speed_importance = 0.2

        if speed < ideal_speed_min:
            speed_ratio = speed / ideal_speed_min
        elif speed <= ideal_speed_max:
            mid_point = (ideal_speed_min + ideal_speed_max) / 2
            speed_diff = abs(speed - mid_point)
            max_diff = (ideal_speed_max - ideal_speed_min) / 2
            speed_ratio = 1.0 - (speed_diff / max_diff) ** 2
        else:
            speed_ratio = ideal_speed_max / speed

        speed_reward = 0.02 * speed_importance * (speed_ratio ** 1.5)  # 从0.04降到0.02
        total_reward += speed_reward
        # 5. 移动效率奖励 - 降低
        movement_reward = 0
        if speed > 10:
            movement_direction_rad = math.atan2(self.vy, self.vx)
            target_direction_rad = math.atan2(dy, dx)
            movement_target_diff_rad = (target_direction_rad - movement_direction_rad) % (2 * math.pi)
            if movement_target_diff_rad > math.pi:
                movement_target_diff_rad -= 2 * math.pi

            movement_alignment = math.cos(movement_target_diff_rad)

            if movement_alignment > 0:
                movement_reward = 0.02 * (movement_alignment ** 1.2)  # 从0.04降到0.02

        total_reward += movement_reward
        # 6. 角度对齐奖励 - 降低，但保持激励
        if dist < 800:
            normalized_angle_diff = min(1.0, angle_diff / 180.0)
            angle_alignment_reward = 0.02 * (1 - normalized_angle_diff) ** 2  # 从0.05降到0.02
            total_reward += angle_alignment_reward
        # 7. 智能减速奖励 - 降低
        if dist < 600:
            ideal_speed = max(10, min(70, dist / 10))
            speed_diff = abs(speed - ideal_speed)
            if speed_diff < 20:
                speed_match_ratio = 1.0 - min(1.0, speed_diff / 20)
                deceleration_reward = 0.02 * (speed_match_ratio ** 1.5)  # 从0.04降到0.02
                total_reward += deceleration_reward
        # 8. 时间惩罚 - 保持轻微
        time_penalty = -0.002
        total_reward += time_penalty
        return max(-0.05, min(0.15, total_reward))  # 进一步压缩奖励范围

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        print("二次训练服务器已关闭")


# 主训练函数
def main():
    # 创建环境
    env = StarSectorEnv(max_steps=MAX_STEPS)
    state_dim = 30
    action_dim = 8

    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE,
        buffer_capacity=BUFFER_CAPACITY
    )

    # 加载之前训练的模型
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"加载模型: {LOAD_MODEL_PATH}")
        try:
            # 使用DQNAgent的load_checkpoint方法
            loaded_episode = agent.load_checkpoint(LOAD_MODEL_PATH)
            print(f"模型加载成功，从第{loaded_episode}轮继续训练")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将从头开始训练")
    else:
        print(f"模型文件不存在: {LOAD_MODEL_PATH}")
        print("将从头开始训练")

    # 训练统计
    episode_rewards = []
    episode_success = []
    tactical_usage = []

    print("开始二次训练...")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"学习率: {LEARNING_RATE}, 初始探索率: {EPSILON_START}")

    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            success = False
            tactical_count = 0

            for step in range(MAX_STEPS):
                # 选择动作
                action = agent.select_action(state)

                # 统计战术使用
                if action == 7:  # 假设7是战术动作
                    tactical_count += 1

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                agent.push_memory(state, action, reward, next_state, done)

                # 训练
                loss = agent.train()

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
            tactical_usage.append(tactical_count / MAX_STEPS * 100)  # 战术使用率百分比

            # 每10轮输出一次统计
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                success_rate = np.mean(episode_success[-10:]) * 100
                avg_tactical = np.mean(tactical_usage[-10:])

                print(f"\nEpisode {episode}/{EPISODES} | "
                      f"Reward: {total_reward:.1f} (Avg: {avg_reward:.1f}) | "
                      f"Success: {success} | Success Rate: {success_rate:.1f}% | "
                      f"Tactical Usage: {avg_tactical:.1f}% | "
                      f"Epsilon: {agent.epsilon:.3f}")

            # 每100轮保存一次模型
            if episode % 100 == 0:
                if not os.path.exists("models_second"):
                    os.makedirs("models_second")
                save_path = f"models_second/dqn_second_model_try_episode_{episode}.pth"

                # 使用DQNAgent的save方法
                agent.save(save_path, episode)
                print(f"二次训练模型已保存: {save_path}")

            env.now_episode += 1

    except KeyboardInterrupt:
        print("二次训练被用户中断")
    finally:
        env.close()

        # 保存最终模型
        final_save_path = "models_second/dqn_second_final_model.pth"
        agent.save(final_save_path, EPISODES)
        print(f"二次训练完成，最终模型已保存为: {final_save_path}")


if __name__ == "__main__":
    main()
