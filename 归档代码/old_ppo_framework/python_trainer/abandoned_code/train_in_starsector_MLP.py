import time
import os
import math
import random
import numpy as np
import torch
import socket
from abandoned_code.dqn_agent_MLP import DQNAgent

# 训练参数
EPISODES = 2000
MAX_STEPS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99
TARGET_UPDATE = 200
BUFFER_CAPACITY = 100000
MAX_SPEED = 350
MAX_ANGULAR_VELOCITY = 20.0
RAY_MAX_DISTANCE = 1


class StarSectorEnv:
    """简化版游戏环境"""

    def __init__(self, max_steps=1000):
        # Socket服务器初始化
        self.host = '127.0.0.1'
        self.port = 65432
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 设置socket选项，重用地址和增大缓冲区
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # 64KB发送缓冲区
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # 64KB接收缓冲区

        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"DQN训练服务器已启动，监听 {self.host}:{self.port}")

        # 等待连接
        self.conn, self.addr = self.sock.accept()

        # 设置连接的缓冲区
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        print(f"来自 {self.addr} 的连接已建立")
        self.conn.setblocking(True)

        # 输出缓冲区信息
        snd_buf = self.conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        rcv_buf = self.conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(f"Socket缓冲区 - 发送: {snd_buf} bytes, 接收: {rcv_buf} bytes")

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

    def reset(self):
        """重置环境 - 基于原来的成功版本"""
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
        """从游戏端接收完整状态，添加数据验证"""
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
                if not value.strip():  # 空值检查
                    print(f"警告: 第{i}个值为空")
                    values[i] = "0"

            # 解析状态数据，添加异常处理
            try:
                self.x = self._safe_float_parse(values[0])
                self.y = self._safe_float_parse(values[1])
                self.angle = self._safe_float_parse(values[2])
                self.vx = self._safe_float_parse(values[3])
                self.vy = self._safe_float_parse(values[4])
                self.vr = self._safe_float_parse(values[5])

                # 解析射线数据
                self.rays = [self._safe_float_parse(values[i]) for i in range(6, 22)]

                # 解析战术系统状态
                self.tactical_cooldown = int(float(values[22]) if values[22].strip() else 0)
                self.tactical_available = int(float(values[23]) if values[23].strip() else 0)
                self.tactical_active = int(float(values[24]) if values[24].strip() else 0)

            except Exception as e:
                print(f"数据解析错误: {e}, 原始数据: {decoded}")
                # 使用默认值
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
            # 检查是否为NaN或无穷大
            if math.isnan(result) or math.isinf(result):
                print(f"警告: 检测到异常数值: {value}")
                return 0.0
            return result
        except (ValueError, TypeError):
            print(f"警告: 无法解析浮点数: {value}")
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

    def _get_state(self):
        """获取28维状态向量，添加边界检查"""
        try:
            # 验证基础数据
            if (math.isnan(self.x) or math.isnan(self.y) or
                    math.isnan(self.vx) or math.isnan(self.vy) or
                    math.isnan(self.vr)):
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

            # 归一化（添加边界检查）
            max_dist = 20000
            relative_x_norm = max(-10.0, min(10.0, relative_x / (max_dist / 2)))
            relative_y_norm = max(-10.0, min(10.0, relative_y / (max_dist / 2)))

            # 角度使用sin/cos表示
            ship_angle_rad = math.radians(self.angle)
            target_angle_rad = math.radians(self.target_angle)

            # 计算目标相对于飞船的方向
            target_direction_rad = math.atan2(dy, dx)  # 目标的绝对方向
            relative_direction_rad = target_direction_rad - ship_angle_rad  # 相对于飞船的方向

            # 归一化到[-π, π]
            while relative_direction_rad > math.pi:
                relative_direction_rad -= 2 * math.pi
            while relative_direction_rad < -math.pi:
                relative_direction_rad += 2 * math.pi

            # 计算相对方向的sin/cos
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

            # 最终状态验证
            for i, val in enumerate(state):
                if math.isnan(val) or math.isinf(val):
                    print(f"警告: 状态向量第{i}维异常: {val}")
                    state[i] = 0.0

            # 使用你原来的简洁版本，添加相对方向信息
            rel_x, rel_y = state[0], state[1]
            vx_norm, vy_norm = state[2], state[3]
            sin_rel_dir, cos_rel_dir = state[9], state[10]  # 新增的相对方向
            tactical_cd, tactical_avail, tactical_active = state[-3], state[-2], state[-1]

            # 计算实际距离和速度
            actual_dist = current_dist
            actual_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

            # 计算相对方向的角度（度数）
            relative_direction_deg = math.degrees(relative_direction_rad)

            # 构建简洁的状态预览
            preview_str = f"距离: {actual_dist:6.1f} | 速度: {actual_speed:5.1f} | "
            preview_str += f"相对位置: ({rel_x:5.2f}, {rel_y:5.2f}) | "
            preview_str += f"相对方向: {relative_direction_deg:6.1f}° | "
            #preview_str += f"战术: CD[{tactical_cd:1.0f}] Avail[{tactical_avail:1.0f}] Active[{tactical_active:1.0f}]"

            # 使用回车符实现不滚动刷新
            print(f"\r{preview_str}", end="", flush=True)


            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            # 返回安全的状态
            return np.zeros(28, dtype=np.float32)

    def _get_basic_reward(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        # 1. 到达目标奖励 - 保持合理
        if dist < 150:
            base_reward = 30.0  # 适中基础奖励
            speed_bonus = 15 * max(0, 1 - min(speed, 10) / 10)
            time_bonus = 15 * (1 - self.steps / MAX_STEPS)
            arrival_reward = base_reward + speed_bonus + time_bonus

            self.success = True
            self._reset_target()
            return arrival_reward
        total_reward = 0
        # 2. 绝对距离奖励 - 适度提供基线引导
        if dist < 3000:  # 3公里内
            absolute_distance_reward = 0.1 * (1 - min(1.0, dist / 3000))
            total_reward += absolute_distance_reward
        # 3. 距离变化奖励 - 平衡设计
        distance_change_reward = 0
        if self.previous_dist_to_target is not None:
            dist_change = self.previous_dist_to_target - dist

            if dist_change > 0:
                # 使用适度的线性奖励
                distance_change_reward = 0.1 * min(1.0, dist_change / 100)
            elif dist_change < -20:  # 明显远离
                distance_change_reward = -0.05 * min(1.0, abs(dist_change) / 100)

        self.previous_dist_to_target = dist
        total_reward += distance_change_reward
        # 4. 方向奖励 - 连续但适度
        target_direction_rad = math.atan2(dy, dx)
        ship_angle_rad = math.radians(self.angle)
        angle_diff_rad = (target_direction_rad - ship_angle_rad) % (2 * math.pi)
        if angle_diff_rad > math.pi:
            angle_diff_rad -= 2 * math.pi

        alignment = math.cos(angle_diff_rad)

        # 适度的连续方向奖励
        continuous_direction_reward = 0.08 * alignment + 0.02  # 范围从-0.06到+0.10
        total_reward += continuous_direction_reward
        # 5. 移动奖励 - 适度提高
        movement_reward = 0
        if speed > 15:  # 有显著速度
            movement_direction_rad = math.atan2(self.vy, self.vx)
            movement_target_diff_rad = (target_direction_rad - movement_direction_rad) % (2 * math.pi)
            if movement_target_diff_rad > math.pi:
                movement_target_diff_rad -= 2 * math.pi

            movement_alignment = math.cos(movement_target_diff_rad)

            if movement_alignment > 0.6:  # 基本方向正确
                movement_reward = 0.05 * movement_alignment  # 最大0.05

        total_reward += movement_reward
        # 6. 旋转奖励/惩罚 - 适度
        rotation_reward = 0
        angle_diff_deg = math.degrees(angle_diff_rad)

        if abs(angle_diff_deg) > 30 and abs(self.vr) > 5:
            correct_rotation = (angle_diff_rad * self.vr) < 0
            if correct_rotation:
                rotation_reward = 0.02 * min(1.0, abs(self.vr) / 20)
            else:
                rotation_reward = -0.02 * min(1.0, abs(self.vr) / 20)

        total_reward += rotation_reward
        # 7. 速度控制奖励 - 适度
        speed_reward = 0
        if dist < 800:
            ideal_speed = max(20, dist / 15)
            if abs(speed - ideal_speed) < 20:  # 速度接近理想值
                speed_reward = 0.01

        total_reward += speed_reward
        # 8. 时间惩罚 - 适度
        time_penalty = -0.005
        total_reward += time_penalty
        # 合理的奖励范围
        return max(-0.2, min(0.3, total_reward))

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        print("训练服务器已关闭")


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

    # 训练统计
    episode_rewards = []
    episode_success = []

    print("开始DQN训练...")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            success = False

            for step in range(MAX_STEPS):
                # 选择动作
                action = agent.select_action(state)

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

            # 每10轮输出一次统计
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                success_rate = np.mean(episode_success[-10:]) * 100

                print(f"Episode {episode}/{EPISODES} | "
                      f"Reward: {total_reward:.1f} (Avg: {avg_reward:.1f}) | "
                      f"Success: {success} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {agent.epsilon:.3f}")

            # 每100轮保存一次模型
            if episode % 100 == 0:
                if not os.path.exists("models"):
                    os.makedirs("models")
                save_path = f"models/dqn_model_episode_{episode}.pth"
                agent.save(save_path, episode)
                print(f"模型已保存: {save_path}")

            env.now_episode += 1

    except KeyboardInterrupt:
        print("训练被用户中断")
    finally:
        env.close()

        # 保存最终模型
        agent.save("dqn_final_model——1.pth")
        print("训练完成，最终模型已保存为 dqn_final_model——1.pth")


if __name__ == "__main__":
    main()