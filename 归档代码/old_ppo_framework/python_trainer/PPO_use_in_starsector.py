import time
import os
import math
import random
import numpy as np
import torch
import socket


class PPOShipController:
    """PPO飞船控制器 - 用于加载训练好的模型并进行推理"""

    def __init__(self, model_path, state_dim=11, action_dim=7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()  # 设置为评估模式

        print(f"PPO模型已加载: {model_path}")
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

    def _load_model(self, model_path):
        """加载PPO模型 - 修复版本，包含完整的网络结构"""

        # 定义完整的网络结构（必须与训练时完全相同）
        class PPONetwork(torch.nn.Module):
            def __init__(self, state_dim, action_dim, hidden_size=256):
                super().__init__()
                self.state_dim = state_dim

                # 共享特征提取层（与训练时相同）
                self.shared = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_size),
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                )

                # Actor - 输出动作概率分布
                self.actor = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size // 2, action_dim)
                )

                # Critic - 输出状态价值（推理时不需要，但必须定义以加载权重）
                self.critic = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size // 2, 1)
                )

            def forward(self, state):
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                features = self.shared(state)
                actor_output = self.actor(features)
                critic_output = self.critic(features)
                return actor_output, critic_output

            def get_action(self, state):
                """推理时只使用Actor部分"""
                actor_output, _ = self.forward(state)
                return actor_output

        # 创建网络并加载权重
        model = PPONetwork(self.state_dim, self.action_dim)
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载完整的权重（包括Actor和Critic）
        model.load_state_dict(checkpoint['policy_state_dict'])
        model.to(self.device)

        return model

    def get_action(self, state):
        """根据状态获取动作"""
        with torch.no_grad():  # 不计算梯度，提高推理速度
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)

            # 获取动作logits（只使用Actor部分）
            logits = self.model.get_action(state_tensor)

            # 使用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)

            # 选择概率最高的动作（贪婪策略）
            action = torch.argmax(probs, dim=-1).item()

            # 调试信息：显示动作概率分布
            if random.random() < 0.01:  # 1%的概率显示调试信息
                probs_np = probs.cpu().numpy()[0]
                print(f"\n动作概率: {[f'{p:.3f}' for p in probs_np]}, 选择动作: {action}")

            return action


class StarSectorInferenceEnv:
    """游戏推理环境 - 只进行推理，不训练"""

    def __init__(self, model_path, max_steps=1000):
        # 加载PPO控制器
        self.controller = PPOShipController(model_path)

        # Socket服务器初始化
        self.host = '127.0.0.1'
        self.port = 65432
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 设置socket选项
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"PPO推理服务器已启动，监听 {self.host}:{self.port}")

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

        # 推理状态
        self.steps = 0
        self.max_steps = max_steps
        self.success = False

        # 目标点刷新计时器
        self.last_target_refresh = time.time()
        self.target_refresh_interval = 10.0  # 每10秒刷新一次目标点

        # 动作映射（与训练时一致）
        self.action_names = {
            0: "无操作",
            1: "前进(W)",
            2: "后退(S)",
            3: "左转(A)",
            4: "右转(D)",
            5: "左平移(Q)",
            6: "右平移(E)"
        }

    def reset(self):
        """重置环境"""
        # 发送重置指令
        self._send_action(100)
        # 接收初始状态
        self._receive_full_state()
        # 重置状态
        self.steps = 0
        self.success = False
        # 设置初始目标点
        self._reset_target()
        # 重置目标点刷新计时器
        self.last_target_refresh = time.time()
        return self._get_state()

    def run_inference(self):
        """运行推理循环"""
        print("开始PPO推理...")
        print("按 Ctrl+C 停止推理")

        try:
            state = self.reset()
            episode_count = 0
            success_count = 0

            while True:
                # 使用PPO模型选择动作
                action = self.controller.get_action(state)

                # 执行动作
                next_state, _, done, _ = self.step(action)

                # 显示当前状态和动作
                self._display_status(state, action)

                state = next_state
                self.steps += 1

                if done:
                    episode_count += 1
                    if self.success:
                        success_count += 1

                    success_rate = (success_count / episode_count) * 100 if episode_count > 0 else 0
                    print(f"\n回合 {episode_count} 结束! 步骤: {self.steps}, 成功: {self.success}")
                    print(f"总成功率: {success_rate:.1f}% ({success_count}/{episode_count})")

                    # 重置环境继续下一个回合
                    state = self.reset()

                # 添加小的延迟，避免过快循环
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n推理被用户中断")
        except Exception as e:
            print(f"\n推理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.close()

    def _reset_target(self):
        """重置目标位置 - 随机生成新的目标点"""
        # 随机生成目标位置，确保不会太近或太远
        min_dist = 500
        max_dist = 2000

        # 随机角度和距离
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_dist, max_dist)

        # 以飞船当前位置为参考点生成目标
        self.target_x = 0 + distance * math.cos(angle)
        self.target_y = 1500 + distance * math.sin(angle)
        self.target_angle = random.uniform(0, 360)

        print(f"新目标点: ({self.target_x:.1f}, {self.target_y:.1f}), 角度: {self.target_angle:.1f}°")

    def step(self, action):
        """执行动作"""
        # 接收新状态
        self._receive_full_state()

        # 检查是否成功（简单判断距离）
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 150:
            self.success = True

        # 检查结束条件
        done = self.steps >= self.max_steps or self.success

        # 发送动作响应
        self._send_action(action)

        return self._get_state(), 0, done, {}

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
                print(f"警告: 期望25个值，但收到 {len(values)} 个值")
                # 填充缺失的值
                while len(values) < 25:
                    values.append("0")

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

                # 解析激光数据（16个值）
                self.rays = [self._safe_float_parse(values[i]) for i in range(6, 22)]

                # 解析战术系统状态（3个值）
                self.tactical_cooldown = int(float(values[22]) if values[22].strip() else 0)
                self.tactical_available = int(float(values[23]) if values[23].strip() else 0)
                self.tactical_active = int(float(values[24]) if values[24].strip() else 0)

            except Exception as e:
                print(f"数据解析错误: {e}")
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
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ValueError, TypeError):
            return 0.0

    def _get_state(self):
        """获取状态向量（与训练时相同的11维状态）"""
        try:
            # 计算相对坐标
            dx = self.target_x - self.x
            dy = self.target_y - self.y

            # 限制距离范围
            max_safe_dist = 50000
            current_dist = math.sqrt(dx ** 2 + dy ** 2)
            if current_dist > max_safe_dist:
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

            # 构建状态向量（11维，与训练时相同）
            state = [
                relative_x_norm, relative_y_norm,
                max(-5.0, min(5.0, self.vx / 350)),
                max(-5.0, min(5.0, self.vy / 350)),
                max(-5.0, min(5.0, self.vr / 20.0)),
                math.sin(ship_angle_rad), math.cos(ship_angle_rad),
                math.sin(target_angle_rad), math.cos(target_angle_rad),
                math.sin(relative_direction_rad), math.cos(relative_direction_rad)
            ]

            # 状态验证
            for i, val in enumerate(state):
                if math.isnan(val) or math.isinf(val):
                    state[i] = 0.0

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"状态计算错误: {e}")
            return np.zeros(11, dtype=np.float32)

    def _display_status(self, state, action):
        """显示当前状态和动作"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        # 获取相对方向角度
        relative_direction_deg = math.degrees(math.atan2(state[10], state[9]))

        # 计算距离下次刷新还有多少秒
        time_until_refresh = max(0, self.target_refresh_interval - (time.time() - self.last_target_refresh))

        status_str = (f"步骤: {self.steps:4d} | "
                      f"距离: {dist:6.1f} | "
                      f"速度: {speed:5.1f} | "
                      f"相对方向: {relative_direction_deg:6.1f}° | "
                      f"动作: {self.action_names[action]} | "
                      f"刷新倒计时: {time_until_refresh:4.1f}s")

        print(f"\r{status_str}", end="", flush=True)

    def close(self):
        """关闭环境"""
        self.conn.close()
        self.sock.close()
        print("\n推理服务器已关闭")


def main():
    # 模型路径 - 修改为你的实际模型路径
    model_path = "previous_models/ppo_real_model_final——1.pth"

    # 如果默认路径不存在，尝试其他可能的位置
    if not os.path.exists(model_path):
        possible_paths = [
            "ppo_real_model_final——1.pth",
            "models/ppo_real_model_final——1.pth",
            "ppo_ship_model_final.pth",
            "models/ppo_ship_model_final.pth"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件")
        print("请确保模型文件存在于以下位置之一:")
        print(" - ppo_real_model_final——1.pth")
        print(" - models/ppo_real_model_final——1.pth")
        print(" - ppo_ship_model_final.pth")
        print(" - models/ppo_ship_model_final.pth")
        print("\n或者修改main()函数中的model_path变量")
        return

    print(f"使用模型: {model_path}")

    # 创建推理环境并运行
    env = StarSectorInferenceEnv(model_path)
    env.run_inference()


if __name__ == "__main__":
    main()