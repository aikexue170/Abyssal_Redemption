import pygame
import math
import numpy as np
import random


class SimpleShipEnv:
    def __init__(self, screen_width=800, screen_height=600):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Ship Docking Simulator")

        # 物理参数
        self.max_speed = 350
        self.max_angular_velocity = 20
        self.acceleration = 5  # 前进加速度
        self.deceleration = 3  # 后退加速度
        self.strafe_acceleration = 2.5  # 平移加速度
        self.angular_acceleration = 5  # 转向加速度
        self.friction = 0.98  # 速度衰减
        self.angular_friction = 0.96  # 角速度衰减

        # 飞船状态
        self.reset()

        # 颜色
        self.ship_color = (0, 255, 255)
        self.target_color = (255, 0, 0)
        self.trail_color = (100, 100, 255, 50)

        # 轨迹记录
        self.trail = []
        self.max_trail_length = 100

        # 字体
        self.font = pygame.font.Font(None, 24)

    def reset(self):
        """重置环境"""
        # 飞船初始位置和状态
        self.x = self.screen_width // 2
        self.y = self.screen_height // 2
        self.angle = 0  # 角度，0度指向右
        self.vx = 0
        self.vy = 0
        self.vr = 0  # 角速度

        # 目标位置
        self._reset_target()

        # 训练状态
        self.steps = 0
        self.success = False
        self.previous_dist_to_target = None

        # 清空轨迹
        self.trail = []

        return self._get_state()

    def _reset_target(self):
        """重置目标位置"""
        # 随机生成目标位置，确保不会太近或太远
        min_dist = 100
        max_dist = 300

        while True:
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(min_dist, max_dist)

            self.target_x = self.x + distance * math.cos(angle)
            self.target_y = self.y + distance * math.sin(angle)
            self.target_angle = random.uniform(0, 360)

            # 确保目标在屏幕内
            if (0 <= self.target_x <= self.screen_width and
                    0 <= self.target_y <= self.screen_height):
                break

    def step(self, action):
        """执行动作"""
        self.steps += 1

        # 应用物理
        self._apply_action(action)
        self._update_physics()

        # 检查边界
        self._check_boundaries()

        # 记录轨迹
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        # 计算奖励
        reward = self._get_reward()

        # 检查结束条件
        done = self.steps >= 1000 or self.success

        # 如果成功，重置目标
        if self.success:
            self._reset_target()
            self.success = False

        return self._get_state(), reward, done, {}

    def _apply_action(self, action):
        """应用动作到飞船"""
        # 动作映射: 0=W(前进), 1=S(后退), 2=A(左转), 3=D(右转), 4=Q(左平移), 5=E(右平移), 6=无动作

        if action == 0:  # W - 前进
            self.vx += self.acceleration * math.cos(math.radians(self.angle))
            self.vy += self.acceleration * math.sin(math.radians(self.angle))
        elif action == 1:  # S - 后退
            self.vx -= self.deceleration * math.cos(math.radians(self.angle))
            self.vy -= self.deceleration * math.sin(math.radians(self.angle))
        elif action == 2:  # A - 左转
            self.vr -= self.angular_acceleration
        elif action == 3:  # D - 右转
            self.vr += self.angular_acceleration
        elif action == 4:  # Q - 左平移
            self.vx += self.strafe_acceleration * math.cos(math.radians(self.angle - 90))
            self.vy += self.strafe_acceleration * math.sin(math.radians(self.angle - 90))
        elif action == 5:  # E - 右平移
            self.vx += self.strafe_acceleration * math.cos(math.radians(self.angle + 90))
            self.vy += self.strafe_acceleration * math.sin(math.radians(self.angle + 90))
        # 动作6什么都不做

    def _update_physics(self):
        """更新物理状态"""
        # 限制速度
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        # 限制角速度
        self.vr = max(-self.max_angular_velocity, min(self.max_angular_velocity, self.vr))

        # 应用摩擦
        self.vx *= self.friction
        self.vy *= self.friction
        self.vr *= self.angular_friction

        # 更新位置和角度
        self.x += self.vx * 0.1  # dt=0.1
        self.y += self.vy * 0.1
        self.angle += self.vr * 0.1

        # 规范化角度
        self.angle %= 360

    def _check_boundaries(self):
        """检查边界，反弹或限制"""
        # 简单的边界反弹
        if self.x < 0:
            self.x = 0
            self.vx = abs(self.vx) * 0.5
        elif self.x > self.screen_width:
            self.x = self.screen_width
            self.vx = -abs(self.vx) * 0.5

        if self.y < 0:
            self.y = 0
            self.vy = abs(self.vy) * 0.5
        elif self.y > self.screen_height:
            self.y = self.screen_height
            self.vy = -abs(self.vy) * 0.5

    def _get_state(self):
        """获取状态向量"""
        # 相对位置
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # 相对角度计算
        target_direction_rad = math.atan2(dy, dx)
        ship_angle_rad = math.radians(self.angle)
        relative_direction_rad = target_direction_rad - ship_angle_rad

        # 规范化到[-π, π]
        while relative_direction_rad > math.pi:
            relative_direction_rad -= 2 * math.pi
        while relative_direction_rad < -math.pi:
            relative_direction_rad += 2 * math.pi

        # 构建状态向量 (简化版，可以根据需要扩展)
        state = [
            # 相对位置 (归一化)
            dx / 400.0,  # 假设最大距离400
            dy / 400.0,

            # 速度 (归一化)
            self.vx / self.max_speed,
            self.vy / self.max_speed,
            self.vr / self.max_angular_velocity,

            # 角度信息
            math.sin(ship_angle_rad),
            math.cos(ship_angle_rad),

            # 相对方向
            math.sin(relative_direction_rad),
            math.cos(relative_direction_rad),

            # 距离信息
            dist / 400.0
        ]

        return np.array(state, dtype=np.float32)

    def _get_reward(self):
        """奖励函数 - 我们可以在这里快速迭代不同的设计"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        # 1. 到达目标奖励
        if dist < 20:  # 成功停靠距离
            base_reward = 50.0
            speed_bonus = 20 * max(0, 1 - min(speed, 20) / 20)  # 低速奖励
            self.success = True
            return base_reward + speed_bonus

        total_reward = 0

        # 2. 距离变化奖励
        if self.previous_dist_to_target is not None:
            dist_change = self.previous_dist_to_target - dist
            if dist_change > 0:
                distance_change_reward = 0.2 * min(1.0, dist_change / 10)
            else:
                distance_change_reward = -0.1 * min(1.0, abs(dist_change) / 10)
            total_reward += distance_change_reward

        self.previous_dist_to_target = dist

        # 3. 方向对齐奖励
        target_direction_rad = math.atan2(dy, dx)
        ship_angle_rad = math.radians(self.angle)
        angle_diff_rad = (target_direction_rad - ship_angle_rad) % (2 * math.pi)
        if angle_diff_rad > math.pi:
            angle_diff_rad -= 2 * math.pi

        alignment = math.cos(angle_diff_rad)
        direction_reward = 0.1 * alignment
        total_reward += direction_reward

        # 4. 速度控制奖励 (接近目标时减速)
        if dist < 100:
            ideal_speed = max(5, dist / 5)
            speed_penalty = -0.05 * min(1.0, abs(speed - ideal_speed) / 20)
            total_reward += speed_penalty

        # 5. 时间惩罚
        time_penalty = -0.01
        total_reward += time_penalty

        return total_reward

    def render(self):
        """渲染环境"""
        self.screen.fill((0, 0, 0))

        # 绘制轨迹
        for i, (trail_x, trail_y) in enumerate(self.trail):
            alpha = int(255 * i / len(self.trail))
            color = (100, 100, 255, alpha)
            pygame.draw.circle(self.screen, color, (int(trail_x), int(trail_y)), 2)

        # 绘制飞船
        ship_points = self._get_ship_points()
        pygame.draw.polygon(self.screen, self.ship_color, ship_points)

        # 绘制飞船方向指示器
        direction_x = self.x + 20 * math.cos(math.radians(self.angle))
        direction_y = self.y + 20 * math.sin(math.radians(self.angle))
        pygame.draw.line(self.screen, (255, 255, 0), (self.x, self.y),
                         (direction_x, direction_y), 2)

        # 绘制目标
        pygame.draw.circle(self.screen, self.target_color,
                           (int(self.target_x), int(self.target_y)), 10)

        # 绘制目标方向指示器
        target_dir_x = self.target_x + 15 * math.cos(math.radians(self.target_angle))
        target_dir_y = self.target_y + 15 * math.sin(math.radians(self.target_angle))
        pygame.draw.line(self.screen, (255, 100, 100),
                         (self.target_x, self.target_y),
                         (target_dir_x, target_dir_y), 2)

        # 绘制信息
        dist = math.sqrt((self.target_x - self.x) ** 2 + (self.target_y - self.y) ** 2)
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        info_text = [
            f"Distance: {dist:.1f}",
            f"Speed: {speed:.1f}",
            f"Steps: {self.steps}",
            f"Angle: {self.angle:.1f}°"
        ]

        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()

    def _get_ship_points(self):
        """获取飞船多边形的顶点"""
        points = []
        length = 20
        width = 10

        # 飞船形状：三角形
        angle_rad = math.radians(self.angle)

        # 船头
        nose_x = self.x + length * math.cos(angle_rad)
        nose_y = self.y + length * math.sin(angle_rad)

        # 左翼
        left_x = self.x + width * math.cos(angle_rad + math.pi / 2)
        left_y = self.y + width * math.sin(angle_rad + math.pi / 2)

        # 右翼
        right_x = self.x + width * math.cos(angle_rad - math.pi / 2)
        right_y = self.y + width * math.sin(angle_rad - math.pi / 2)

        return [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)]

    def close(self):
        pygame.quit()