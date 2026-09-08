"""diag_env.py —— 环境正确性诊断: 用手写 PD 控制器跑导航任务。

如果 PD 能稳定成功 -> 环境与奖励机制没问题, 瓶颈在 RL 调参;
如果 PD 也失败 -> 环境/成功判定有 bug。
同时统计回合结束原因分布 (成功/超时/出界)。
"""

from __future__ import annotations

import torch

from tensor_env import STAGES, NavEnv


def pd_action(env: NavEnv) -> torch.Tensor:
    """船体系 PD: 期望速度指向目标, 油门正比于速度误差。"""
    rel, dist, dh = env._dist_dh()
    th = torch.deg2rad(env.facing)
    cos, sin = torch.cos(th), torch.sin(th)
    rel_f = cos * rel[:, 0] + sin * rel[:, 1]
    rel_r = -sin * rel[:, 0] + cos * rel[:, 1]
    vf = cos * env.vel[:, 0] + sin * env.vel[:, 1]
    vs = -sin * env.vel[:, 0] + cos * env.vel[:, 1]
    # 期望速度: 远处 40 su/s, 近处比例减速 (v_des = min(40, 1.2*dist))
    v_max = torch.clamp(1.2 * dist, max=40.0)
    vd_f = v_max * rel_f / dist
    vd_r = v_max * rel_r / dist
    u_m = ((vd_f - vf) / 20.0).clamp(-1, 1)
    u_s = ((vd_r - vs) / 10.0).clamp(-1, 1)
    # 朝向: 阶段>=2 才对齐目标朝向, 否则不转
    u_t = (dh / 30.0).clamp(-1, 1) * (env.success_angle < 180)
    return torch.stack([u_m, u_t, u_s], dim=-1)


def main():
    torch.manual_seed(0)
    n = 64
    for stage in (0, 2):
        env = NavEnv(num_envs=n, device="cuda", randomize=0.15, seed=42 + stage)
        env.set_stage(**STAGES[stage])
        obs = env.reset()
        n_succ = n_time = n_oob = 0
        dist_end = []
        alive = torch.ones(n, dtype=torch.bool, device="cuda")
        for t in range(env.max_steps):
            a = pd_action(env)
            obs, r, term, trunc, info = env.step(a)
            newly = info["done"] & alive
            if newly.any():
                s = info["success"][newly]
                n_succ += int(s.sum())
                oob = (info["dist"][newly] > env.oob_dist)
                n_oob += int(oob.sum())
                n_time += int((~s & ~oob).sum())
                dist_end += info["dist"][newly].tolist()
                env.reset(newly)
                alive[newly] = False
        import statistics
        print(f"stage {stage}: 成功 {n_succ}/{n}  超时 {n_time}  出界 {n_oob}  "
              f"终距中位 {statistics.median(dist_end) if dist_end else -1:.1f}")


if __name__ == "__main__":
    main()
