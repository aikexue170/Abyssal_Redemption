"""eval_policy.py —— 加载训练存档, 用确定性策略跑一批回合, 画轨迹并统计成功率。

用法:
    python eval_policy.py --ckpt runs/nav_v1/latest.pt
    python eval_policy.py --ckpt runs/nav_v1/latest.pt --episodes 16 --stage 2 --out eval.png
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ppo import PPO
from tensor_env import STAGES, NavEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--stage", type=int, default=None,
                    help="评估用课程阶段; 默认取存档里的 stage")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--randomize", type=float, default=0.15)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    stage = args.stage if args.stage is not None else int(ckpt.get("stage", 0))
    agent = PPO(ckpt["obs_dim"], ckpt["act_dim"], device=args.device)
    agent.net.load_state_dict(ckpt["state_dict"])

    n = args.episodes
    env = NavEnv(num_envs=n, device=args.device, max_steps=600,
                 randomize=args.randomize, seed=args.seed)
    env.set_stage(**STAGES[stage])

    obs = env.reset()
    # 记录轨迹
    traj = torch.zeros(env.max_steps + 1, n, 2)
    fac = torch.zeros(env.max_steps + 1, n)
    tgt = env.target_pos.clone().cpu()
    tgt_fac = env.target_facing.clone().cpu()
    traj[0] = env.pos.cpu()
    fac[0] = env.facing.cpu()

    done_all = torch.zeros(n, dtype=torch.bool, device=args.device)
    success = torch.zeros(n, dtype=torch.bool, device=args.device)
    steps_used = torch.full((n,), env.max_steps, dtype=torch.long)

    for t in range(env.max_steps):
        action, _, _, _ = agent.net.act(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        traj[t + 1] = env.pos.cpu()
        fac[t + 1] = env.facing.cpu()
        newly = info["done"] & ~done_all
        success |= info["success"] & ~done_all
        steps_used[newly.cpu()] = t + 1
        done_all |= info["done"]
        if done_all.all():
            traj = traj[:t + 2]
            fac = fac[:t + 2]
            break

    # ---------------- 统计 ----------------
    final_dist = info["dist"]
    speed = env.vel.norm(dim=-1)
    dh = (env.target_facing - env.facing + 180.0) % 360.0 - 180.0
    print(f"ckpt={args.ckpt} stage={stage} episodes={n}")
    print(f"成功率 {success.float().mean().item():.1%}  "
          f"平均终距 {final_dist.mean().item():.1f} su  "
          f"平均终速 {speed.mean().item():.2f} su/s  "
          f"平均朝向误差 {dh.abs().mean().item():.1f}°")

    # ---------------- 绘图 ----------------
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.reshape(-1)
    for i in range(n):
        ax = axes[i]
        T = int(steps_used[i].item())
        p = traj[:T + 1, i].numpy()
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, color="tab:blue", alpha=0.8)
        ax.plot(p[0, 0], p[0, 1], "o", color="tab:green", ms=5, label="start")
        ax.plot(tgt[i, 0], tgt[i, 1], "*", color="tab:red", ms=12, label="target")
        # 最终朝向箭头 (实) 与目标朝向箭头 (虚)
        fl = 30.0
        fx, fy = p[-1]
        th = math.radians(float(fac[min(T, fac.shape[0] - 1), i]))
        ax.arrow(fx, fy, fl * math.cos(th), fl * math.sin(th),
                 head_width=6, color="tab:blue")
        tht = math.radians(float(tgt_fac[i]))
        ax.arrow(tgt[i, 0], tgt[i, 1], fl * math.cos(tht), fl * math.sin(tht),
                 head_width=6, color="tab:red", alpha=0.6)
        ok = bool(success[i])
        ax.set_title(f"#{i} {'OK' if ok else 'fail'} d={final_dist[i]:.0f} "
                     f"dh={abs(dh[i]):.0f}°", fontsize=9,
                     color="tab:green" if ok else "tab:red")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"eval stage={stage} succ={success.float().mean().item():.0%}")
    fig.tight_layout()
    out = args.out or os.path.splitext(args.ckpt)[0] + "_eval.png"
    fig.savefig(out, dpi=130)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
