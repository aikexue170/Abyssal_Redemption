import time
import os
import math
import random
import numpy as np
import torch
import socket
from abandoned_code.dqn_agent_MLP import DQNAgent

# 使用参数
MAX_STEPS = 1000
MAX_SPEED = 350
MAX_ANGULAR_VELOCITY = 20.0
RAY_MAX_DISTANCE = 1

# 模型加载路径
MODEL_PATH = "models_second/dqn_second_model_episode_600.pth"


class StarSectorEnv:
    """使用模式环境 - 简洁版本"""

    def __init__(self, max_steps=1000):
        # Socket服务器初始化
        self.host = '127.0.0.1'
        self.port = 65432
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"AI导航服务器已启动，监听 {self.host}:{self.port}")

        # 等待连接
        self.conn, self.addr = self.sock.accept()
        self.conn.setblocking(True)
        print(f"来自 {self.addr} 的连接已建立")

        # 状态变量
        self.x, self.y, self.angle = 0, 0, 0
        self.vx, self.vy, self.vr = 0, 0, 0
        self.target_x, self.target_y, self.target_angle = 0, 0, 0
        self.rays = [0] * 16
        self.tactical_cooldown = 0
        self.tactical_available = 0
        self.tactical_active = 0

        # 使用状态
        self.steps = 0
        self.max_steps = max_steps
        self.success = False
        self.next_action = 0
        self.mission_count = 0
        self.success_count = 0

    def reset(self):
        """重置环境"""
        self._send_action(100)
        self._receive_full_state()

        self.steps = 0
        self.success = False
        self._reset_target_use_mode()
        self.mission_count += 1

        print(f"开始任务 #{self.mission_count}")
        return self._get_state()

    def step(self, action):
        """执行动作"""
        self.next_action = action
        self._receive_full_state()

        # 检查是否到达目标
        self._check_success()

        done = self.steps >= self.max_steps or self.success
        self._send_action(self.next_action)
        self.steps += 1

        return self._get_state(), done, {}

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
                self.x = float(values[0])
                self.y = float(values[1])
                self.angle = float(values[2])
                self.vx = float(values[3])
                self.vy = float(values[4])
                self.vr = float(values[5])
                self.rays = [float(values[i]) for i in range(6, 22)]
                self.tactical_cooldown = int(float(values[22]))
                self.tactical_available = int(float(values[23]))
                self.tactical_active = int(float(values[24]))

            except Exception as e:
                print(f"数据解析错误: {e}")
                self.x, self.y, self.angle, self.vx, self.vy, self.vr = 0, 0, 0, 0, 0, 0
                self.rays = [0] * 16
                self.tactical_cooldown, self.tactical_available, self.tactical_active = 0, 0, 0

        except Exception as e:
            print(f"接收状态错误: {e}")
            self.close()

    def _reset_target_use_mode(self):
        """使用模式的目标生成"""
        min_distance = 500
        max_distance = 3000
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_distance, max_distance)

        self.target_x = self.x + distance * math.cos(angle)
        self.target_y = self.y + distance * math.sin(angle)
        self.target_angle = random.uniform(0, 360)

    def _check_success(self):
        """检查是否成功到达目标"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # 计算角度差
        angle_diff = abs((self.angle - self.target_angle + 180) % 360 - 180)

        # 成功条件：距离120米内且角度差15度内
        if dist < 150:
            self.success = True
            self.success_count += 1

            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            print(f"任务 #{self.mission_count} 完成! 速度: {speed:.1f}m/s, 角度误差: {angle_diff:.1f}°")
            print(f"累计成功率: {self.success_count}/{self.mission_count}")

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
                self._reset_target_use_mode()
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

            # 简洁的状态显示
            actual_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            angle_diff = abs((self.angle - self.target_angle + 180) % 360 - 180)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            return np.zeros(30, dtype=np.float32)

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        print(f"\nAI导航服务器已关闭")
        print(f"最终统计: {self.success_count}/{self.mission_count} 成功")


# 主使用函数
def main():
    # 创建环境
    env = StarSectorEnv(max_steps=MAX_STEPS)
    state_dim = 30
    action_dim = 8

    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=0.0,  # 禁用探索
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=1,
        target_update=10000,
        buffer_capacity=1
    )

    # 加载训练好的模型
    if os.path.exists(MODEL_PATH):
        print(f"加载AI模型: {MODEL_PATH}")
        try:
            agent.load_checkpoint(MODEL_PATH)
            agent.epsilon = 0.0
            print("模型加载成功！AI导航已就绪")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return
    else:
        print(f"模型文件不存在: {MODEL_PATH}")
        return

    print("AI导航系统启动成功！按 Ctrl+C 停止")
    print("-" * 40)

    try:
        while True:
            state = env.reset()
            done = False

            while not done:
                # 选择动作（完全利用，无探索）
                action = agent.select_action(state, training=False)

                # 执行动作
                next_state, done, _ = env.step(action)

                # 更新状态
                state = next_state

                # 短暂延迟
                time.sleep(0.01)

            # 任务完成后等待
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n用户中断导航")
    finally:
        env.close()


if __name__ == "__main__":
    main()
