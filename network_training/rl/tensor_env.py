"""tensor_env.py —— 全 GPU 张量化导航环境（临渊号拟合动力学）。

设计要点:
- 动力学与 evaluate_model.py 的 channel_accel_world 严格一致:
  世界系惯性(速度不随船体旋转), 通道力沿船体轴施加,
  施力与否由速度在船体系上的投影决定(分段恒定加速度模型)。
- 所有状态/动作为 [N, ...] 的 CUDA tensor, 单步 = 一次 broadcast 前向。
- 域随机化: 每 env 每通道参数乘 U(1-randomize, 1+randomize), reset 时重采样。
- 课程学习: set_stage() 调整到点判定与朝向要求(由训练脚本按成功率晋级)。
- 单位与游戏一致: 距离 su, 速度 su/s, 角度 度, 角速度 度/s, dt=0.05 (游戏仿真 tick)。

观测 (8 维, 船体系, 归一化):
  [rel_fwd/1000, rel_right/1000, vf/60, vs/60, ang_vel/20,
   sin(dh), cos(dh), dist/1000]
动作: [move, turn, strafe] ∈ [-1, 1], 与 65432 协议 ACT 指令完全一致。
"""

from __future__ import annotations

import json
import math
import os

import torch

# ---------------- 实测拟合参数 (临渊, 裸船机动; output/params.json 的内嵌副本) ----------------

EMBEDDED_PARAMS = {
    "move":  {"accel_per_unit_pos": 13.8020, "accel_per_unit_neg": 13.8185,
              "force_limit_pos": 40.1124, "force_limit_neg": 19.9783,
              "creep_pos": 0.9593, "creep_neg": -1.0024,
              "v_cap_pos": 49.4358, "v_cap_neg": -42.7952,
              "coast_decel_measured": 0.00026},
    "strafe": {"accel_per_unit_pos": 7.5571, "accel_per_unit_neg": 7.5588,
               "force_limit_pos": 14.9694, "force_limit_neg": 14.9694,
               "creep_pos": -0.0003, "creep_neg": -0.0067,
               "v_cap_pos": 15.0724, "v_cap_neg": -15.0725,
               "coast_decel_measured": 0.0141},
    "turn":  {"accel_per_unit_pos": 18.9902, "accel_per_unit_neg": 18.9957,
              "force_limit_pos": 19.7018, "force_limit_neg": 19.7090,
              "creep_pos": 0.0168, "creep_neg": 0.0810,
              "v_cap_pos": 20.0851, "v_cap_neg": -20.0851,
              "coast_decel_measured": 1.0079},
}

CHANNEL_KEYS = ("accel_per_unit_pos", "accel_per_unit_neg",
                "force_limit_pos", "force_limit_neg",
                "creep_pos", "creep_neg", "v_cap_pos", "v_cap_neg")

OBS_DIM = 8
ACT_DIM = 3

# 课程阶段 (train_nav / eval_policy 共用的单一来源):
#   arrival_radius 到点半径, stop_speed 停稳速度上限,
#   success_angle 成功所需朝向误差(度), heading_weight 朝向 shaping 权重
STAGES = [
    # heading_weight 从 0 阶段就开启 (小权重, 仅 <100su 近场生效):
    # 让"对齐朝向"的技能始终在线, 避免终段晋级时策略需要推翻已有习惯
    dict(arrival_radius=40.0, stop_speed=12.0, success_angle=180.0, heading_weight=0.02, hold_time=0.5),
    dict(arrival_radius=20.0, stop_speed=8.0, success_angle=180.0, heading_weight=0.05, hold_time=1.0),
    dict(arrival_radius=12.0, stop_speed=5.0, success_angle=20.0, heading_weight=0.10, hold_time=1.0),
    dict(arrival_radius=8.0, stop_speed=3.5, success_angle=8.0, heading_weight=0.15, hold_time=1.0),
]


def load_params(path: str | None = None) -> dict:
    """优先读 output/params.json (拟合脚本产物), 不存在则用内嵌副本。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "output", "params.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {ch: raw[ch] for ch in ("move", "strafe", "turn")}
    return EMBEDDED_PARAMS


def _channel_accel(v_proj: torch.Tensor, u: torch.Tensor,
                   p: dict[str, torch.Tensor]) -> torch.Tensor:
    """向量化版 evaluate_model.channel_accel_world。
    v_proj: 速度在通道轴上的投影 [N]; u: 油门 [-1,1] [N]; p: 该通道参数(各 [N])。
    返回沿通道方向的标量加速度 [N]。"""
    a = torch.zeros_like(v_proj)
    pos = u > 0
    neg = u < 0
    # u>0: 投影低于 force_limit → 全加速; 超过后 creep (到 cap 为止)
    a = torch.where(pos & (v_proj < p["force_limit_pos"]),
                    p["accel_per_unit_pos"] * u, a)
    a = torch.where(pos & (v_proj >= p["force_limit_pos"]) & (v_proj < p["v_cap_pos"]),
                    p["creep_pos"].expand_as(a), a)
    # u<0: 对称
    a = torch.where(neg & (v_proj > -p["force_limit_neg"]),
                    p["accel_per_unit_neg"] * u, a)
    a = torch.where(neg & (v_proj <= -p["force_limit_neg"]) & (v_proj > p["v_cap_neg"]),
                    p["creep_neg"].expand_as(a), a)
    return a


class NavEnv:
    """导航任务: 从原点附近出发, 机动到目标点并停下(高阶段要求对齐目标朝向)。"""

    def __init__(self,
                 num_envs: int = 2048,
                 params: dict | None = None,
                 device: str = "cuda",
                 dt: float = 0.05,
                 max_steps: int = 400,          # 20 秒
                 target_min_r: float = 200.0,
                 target_max_r: float = 800.0,
                 randomize: float = 0.15,       # 域随机化幅度
                 coast_drag: bool = False,
                 timeout_dist_penalty: float = 0.02,   # 超时惩罚 = 系数 x 剩余距离
                 near_speed_weight: float = 0.002,      # 近场高速惩罚 (鼓励刹车剖面)
                 near_radius: float = 150.0,
                 seed: int = 0):
        self.N = num_envs
        self.device = torch.device(device)
        self.dt = dt
        self.max_steps = max_steps
        self.target_min_r = target_min_r
        self.target_max_r = target_max_r
        self.randomize = randomize
        self.coast_drag = coast_drag
        self.timeout_dist_penalty = timeout_dist_penalty
        self.near_speed_weight = near_speed_weight
        self.near_radius = near_radius
        self.g = torch.Generator(device=self.device)
        self.g.manual_seed(seed)

        params = params or load_params()
        # 基础参数表: {channel: {key: base float}}; coast_decel 常量化
        self.base = {ch: {k: float(params[ch][k]) for k in CHANNEL_KEYS}
                     for ch in ("move", "strafe", "turn")}
        self.coast_decel = {ch: float(params[ch].get("coast_decel_measured", 0.0))
                            for ch in ("move", "strafe", "turn")}

        # 课程阶段参数 (由 set_stage 调整)
        self.arrival_radius = 12.0     # 到点半径 (su)
        self.stop_speed = 6.0          # 停稳速度上限 (su/s)
        self.success_angle = 180.0     # 成功所需朝向误差 (度); 180 = 不要求
        self.heading_weight = 0.0      # 朝向 shaping 权重
        self.stop_spin = 3.0           # 停稳角速度上限 (度/s): "停下"包含不转
        self.spin_penalty = 0.005      # 角速度惩罚系数 (防直升机式邪道策略)

        # 成功需持续驻留的步数 (1 秒)
        self.hold_steps = int(1.0 / dt)
        # 成功一次性大奖; 成功后回合不终止 (否则停在目标点的驻留价值流 > 大奖,
        # 理性策略会刻意拒绝成功)
        self.success_bonus = 50.0
        # 逃远判定
        self.oob_dist = 2000.0

        d = self.device
        N = self.N
        self.pos = torch.zeros(N, 2, device=d)
        self.vel = torch.zeros(N, 2, device=d)
        self.facing = torch.zeros(N, device=d)          # 度, [0,360)
        self.ang_vel = torch.zeros(N, device=d)         # 度/s
        self.target_pos = torch.zeros(N, 2, device=d)
        self.target_facing = torch.zeros(N, device=d)
        self.prev_dist = torch.zeros(N, device=d)
        self.step_count = torch.zeros(N, dtype=torch.long, device=d)
        self.hold_count = torch.zeros(N, dtype=torch.long, device=d)
        self.success_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        # 域随机化后的每-env参数: {channel: {key: [N]}}
        self.p = {ch: {k: torch.full((N,), v, device=d) for k, v in kv.items()}
                  for ch, kv in self.base.items()}

        self.reset()

    # ---------------- 内部工具 ----------------

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.rand(shape, generator=self.g, device=self.device) * (high - low) + low

    def _randomize_params(self, idx: torch.Tensor):
        if self.randomize <= 0:
            return
        for ch in ("move", "strafe", "turn"):
            scale = 1.0 + self._rand((len(idx),), -self.randomize, self.randomize)
            for k in CHANNEL_KEYS:
                self.p[ch][k][idx] = self.base[ch][k] * scale

    def _dist_dh(self):
        rel = self.target_pos - self.pos
        dist = rel.norm(dim=-1).clamp_min(1e-6)
        dh = (self.target_facing - self.facing + 180.0) % 360.0 - 180.0
        return rel, dist, dh

    def _obs(self) -> torch.Tensor:
        rel, dist, dh = self._dist_dh()
        th = torch.deg2rad(self.facing)
        cos, sin = torch.cos(th), torch.sin(th)
        rel_fwd = cos * rel[:, 0] + sin * rel[:, 1]     # 世界向量 → 船体系
        rel_right = -sin * rel[:, 0] + cos * rel[:, 1]
        vf = cos * self.vel[:, 0] + sin * self.vel[:, 1]
        vs = -sin * self.vel[:, 0] + cos * self.vel[:, 1]
        dhr = torch.deg2rad(dh)
        return torch.stack([
            rel_fwd / 1000.0, rel_right / 1000.0,
            vf / 60.0, vs / 60.0,
            self.ang_vel / 20.0,
            torch.sin(dhr), torch.cos(dhr),
            dist / 1000.0,
        ], dim=-1)

    # ---------------- 课程 ----------------

    def set_stage(self, arrival_radius: float, stop_speed: float,
                  success_angle: float, heading_weight: float,
                  hold_time: float = 1.0):
        self.arrival_radius = arrival_radius
        self.stop_speed = stop_speed
        self.success_angle = success_angle
        self.heading_weight = heading_weight
        self.hold_steps = max(1, int(hold_time / self.dt))

    # ---------------- gym 风格接口 ----------------

    def reset(self, mask: torch.Tensor | None = None):
        """mask=None 全量重置; 否则只重置 mask 为 True 的 env。返回观测。"""
        if mask is None:
            idx = torch.arange(self.N, device=self.device)
        else:
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return self._obs()
        n = len(idx)
        self.pos[idx] = 0.0
        self.vel[idx] = 0.0
        self.facing[idx] = self._rand((n,), 0.0, 360.0)
        self.ang_vel[idx] = 0.0
        ang = self._rand((n,), 0.0, 2.0 * math.pi)
        rad = self._rand((n,), self.target_min_r, self.target_max_r)
        self.target_pos[idx, 0] = rad * torch.cos(ang)
        self.target_pos[idx, 1] = rad * torch.sin(ang)
        self.target_facing[idx] = self._rand((n,), 0.0, 360.0)
        self.step_count[idx] = 0
        self.hold_count[idx] = 0
        self.success_awarded[idx] = False
        self._randomize_params(idx)
        _, dist, _ = self._dist_dh()
        self.prev_dist[idx] = dist[idx]
        return self._obs()

    def step(self, action: torch.Tensor):
        """action: [N,3] ∈ [-1,1] = (move, turn, strafe)。
        返回 obs, reward, terminated, truncated, info。
        info['terminal_obs']: done env 的真实终态观测 (PPO 截断自举用)。"""
        u = action.clamp(-1.0, 1.0)
        dt = self.dt
        th = torch.deg2rad(self.facing)
        cos, sin = torch.cos(th), torch.sin(th)
        fwd_x, fwd_y = cos, sin                 # 船头方向(世界系)
        right_x, right_y = -sin, cos            # 右舷方向(世界系)

        vf = self.vel[:, 0] * fwd_x + self.vel[:, 1] * fwd_y
        vs = self.vel[:, 0] * right_x + self.vel[:, 1] * right_y

        # move 通道 (u[:,0])
        a = _channel_accel(vf, u[:, 0], self.p["move"])
        if self.coast_drag:
            idle = (u[:, 0] == 0) & (a == 0) & (vf.abs() > 1e-6)
            a = torch.where(idle, -self.coast_decel["move"] * torch.sign(vf), a)
        self.vel[:, 0] += a * fwd_x * dt
        self.vel[:, 1] += a * fwd_y * dt

        # strafe 通道 (u[:,2])
        a = _channel_accel(vs, u[:, 2], self.p["strafe"])
        if self.coast_drag:
            idle = (u[:, 2] == 0) & (a == 0) & (vs.abs() > 1e-6)
            a = torch.where(idle, -self.coast_decel["strafe"] * torch.sign(vs), a)
        self.vel[:, 0] += a * right_x * dt
        self.vel[:, 1] += a * right_y * dt

        # turn 通道 (u[:,1]): 船体系角速度, 更新后 clamp 到 cap
        aw = _channel_accel(self.ang_vel, u[:, 1], self.p["turn"])
        if self.coast_drag:
            idle = (u[:, 1] == 0) & (aw == 0) & (self.ang_vel.abs() > 1e-6)
            aw = torch.where(idle, -self.coast_decel["turn"] * torch.sign(self.ang_vel), aw)
        self.ang_vel = (self.ang_vel + aw * dt).clamp(
            self.p["turn"]["v_cap_neg"], self.p["turn"]["v_cap_pos"])
        self.facing = (self.facing + self.ang_vel * dt) % 360.0

        self.pos += self.vel * dt
        self.step_count += 1

        # ---------------- 奖励与终止 ----------------
        _, dist, dh = self._dist_dh()
        speed = self.vel.norm(dim=-1)

        reward = 0.1 * (self.prev_dist - dist)          # 距离进展 shaping
        reward -= 0.005                                  # 时间惩罚
        reward -= self.spin_penalty * self.ang_vel.abs() / 20.0   # 角速度惩罚

        in_zone = dist < self.arrival_radius
        stopped = speed < self.stop_speed
        spin_stopped = self.ang_vel.abs() < self.stop_spin
        aligned = dh.abs() < self.success_angle
        ok = in_zone & stopped & spin_stopped & aligned

        reward = torch.where(in_zone, reward + 0.2, reward)   # 驻留奖励
        if self.near_speed_weight > 0:                         # 近场高速惩罚
            near2 = dist < self.near_radius
            reward = torch.where(near2, reward - self.near_speed_weight * speed, reward)
        if self.heading_weight > 0:                            # 朝向 shaping (近场)
            near = dist < 100.0
            reward = torch.where(near,
                                 reward + self.heading_weight * torch.cos(torch.deg2rad(dh)),
                                 reward)

        self.hold_count = torch.where(ok, self.hold_count + 1,
                                      torch.zeros_like(self.hold_count))
        # 首次达成驻留: 一次性大奖, 回合继续 (不终止)
        just_awarded = (self.hold_count >= self.hold_steps) & ~self.success_awarded
        reward = torch.where(just_awarded, reward + self.success_bonus, reward)
        self.success_awarded |= just_awarded
        success = self.success_awarded          # 本回合是否曾达成 (统计用)

        oob = dist > self.oob_dist
        reward = torch.where(oob, reward - 5.0, reward)

        terminated = oob
        truncated = (self.step_count >= self.max_steps) & ~terminated
        # 超时未完成: 按剩余距离给终局惩罚, 提供指向目标的直接梯度
        reward = torch.where(truncated,
                             reward - self.timeout_dist_penalty * dist, reward)
        done = terminated | truncated

        terminal_obs = self._obs()                  # done env 的终态观测
        self.prev_dist = dist

        info = {
            "success": success,
            "done": done,
            "dist": dist,
            "terminal_obs": terminal_obs,
        }
        return self._obs(), reward, terminated, truncated, info
