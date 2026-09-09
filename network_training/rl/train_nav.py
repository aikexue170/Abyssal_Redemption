"""train_nav.py —— 导航任务训练入口 (全 GPU PPO + 课程学习)。

用法 (本机 conda starsector 环境):
    cd network_training/rl
    python train_nav.py                      # 默认 2048 envs
    python train_nav.py --num-envs 4096 --iters 400 --run-name nav_v1

课程阶段 (滚动成功率 > --advance-threshold 自动晋级):
    0: 到点停下即可 (半径 12, 速度<6)
    1: + 朝向对齐 ±20° (半径 10, 速度<5)
    2: + 朝向对齐 ±8°  (半径 8, 速度<3.5)

输出: runs/<run_name>/ 下 latest.pt (每 --save-every 轮), config.json, log 行。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ppo import PPO
from tensor_env import ACT_DIM, OBS_DIM, STAGES, NavEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=256, help="每轮 rollout 步数")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=1e-5,
                   help="训练结束时学习率 (全程线性衰减)")
    p.add_argument("--time-penalty", type=float, default=0.005)
    p.add_argument("--align-far-weight", type=float, default=0.05)
    p.add_argument("--tail-penalty-weight", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--ent-decay", type=float, default=0.7,
                   help="每次课程晋级时熵系数乘以此衰减 (精密驻留需要策略收敛)")
    p.add_argument("--init-log-std", type=float, default=-0.3)
    p.add_argument("--min-std", type=float, default=0.12, help="策略标准差下限")
    p.add_argument("--randomize", type=float, default=0.15, help="域随机化幅度")
    p.add_argument("--max-steps", type=int, default=600, help="单回合步数 (30s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--advance-threshold", type=float, default=0.7)
    p.add_argument("--advance-min-episodes", type=int, default=4000,
                   help="晋级所需最少回合数 (防早期虚高)")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None, help="checkpoint 路径")
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or time.strftime("nav_%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    torch.manual_seed(args.seed)
    env = NavEnv(num_envs=args.num_envs, device=args.device, dt=0.05,
                 max_steps=args.max_steps, randomize=args.randomize,
                 time_penalty=args.time_penalty,
                 align_far_weight=args.align_far_weight,
                 tail_penalty_weight=args.tail_penalty_weight, seed=args.seed)
    stage = args.start_stage
    env.set_stage(**STAGES[stage])

    agent = PPO(OBS_DIM, ACT_DIM, device=args.device, lr=args.lr,
                gamma=args.gamma, ent_coef=args.ent_coef,
                init_log_std=args.init_log_std, min_std=args.min_std, seed=args.seed)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        agent.net.load_state_dict(ckpt["state_dict"])
        print(f"[resume] {args.resume}")
    agent.init_buffer(args.num_steps, args.num_envs, OBS_DIM, ACT_DIM)

    obs = env.reset()
    # 统计 (滚动)
    ep_ret = torch.zeros(args.num_envs, device=args.device)
    ep_count = 0
    succ_count = 0
    stage_ep_count = 0
    final_dist_sum = 0.0
    ret_hist: list[float] = []
    succ_hist: list[bool] = []
    dist_hist: list[float] = []
    best_succ = -1.0               # 最优成功率 (best.pt 保护, 防崩溃丢好状态)

    t0 = time.time()
    global_steps = 0
    print(f"run={run_name} envs={args.num_envs} steps/iter={args.num_steps} "
          f"batch={args.num_envs * args.num_steps} device={args.device}")
    for it in range(1, args.iters + 1):
        # 学习率线性衰减
        lr_now = args.lr + (args.lr_final - args.lr) * (it - 1) / max(args.iters - 1, 1)
        for g in agent.opt.param_groups:
            g["lr"] = lr_now
        for t in range(args.num_steps):
            action, logp, value, raw = agent.net.act(obs)
            next_obs, rew, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            b = agent.buf
            b["obs"][t] = obs
            b["act"][t] = raw          # 存未 clamp 原样本, 与 logp 一致
            b["logp"][t] = logp
            b["rew"][t] = rew
            b["done"][t] = done.float()
            b["trunc"][t] = truncated.float()
            b["val"][t] = value
            # 截断回合: 记录终态观测价值用于自举
            if truncated.any():
                tv = agent.net.value(info["terminal_obs"][truncated])
                b["tval"][t][truncated] = tv

            ep_ret += rew
            if done.any():
                idxs = done.nonzero(as_tuple=False).squeeze(-1)
                ep_count += len(idxs)
                stage_ep_count += len(idxs)
                succ = info["success"][idxs]
                succ_count += int(succ.sum().item())
                fdist = info["dist"][idxs]
                final_dist_sum += float(fdist.sum().item())
                ret_hist.extend(ep_ret[idxs].tolist())
                succ_hist.extend(succ.tolist())
                dist_hist.extend(fdist.tolist())
                ret_hist, succ_hist, dist_hist = (ret_hist[-5000:], succ_hist[-5000:],
                                                  dist_hist[-5000:])
                ep_ret[idxs] = 0.0
                env.reset(done)

            obs = next_obs
        global_steps += args.num_steps * args.num_envs

        next_value = agent.net.value(obs)
        adv, ret = agent.compute_returns(next_value)
        stats = agent.update(adv, ret)

        # 滚动指标
        win = min(len(succ_hist), 1000)
        succ_rate = sum(succ_hist[-win:]) / max(win, 1)
        mean_ret = sum(ret_hist[-win:]) / max(len(ret_hist[-win:]), 1)
        mean_dist = sum(dist_hist[-win:]) / max(len(dist_hist[-win:]), 1)
        sps = int(global_steps / (time.time() - t0))

        print(f"iter {it:4d} | stage {stage} | ret {mean_ret:8.2f} | "
              f"succ {succ_rate:5.1%} | dist {mean_dist:7.1f} | "
              f"kl {stats['kl']:.4f} | ent {stats['ent']:.3f} | "
              f"vf {stats['vf']:.4f} | lr {lr_now:.2e} | SPS {sps}", flush=True)

        # 课程晋级
        if (stage < len(STAGES) - 1 and succ_rate > args.advance_threshold
                and stage_ep_count >= args.advance_min_episodes):
            stage += 1
            stage_ep_count = 0
            succ_hist.clear()
            env.set_stage(**STAGES[stage])
            agent.ent_coef = max(agent.ent_coef * args.ent_decay, 2e-4)
            print(f"[curriculum] 晋级到 stage {stage}: {STAGES[stage]} "
                  f"ent_coef -> {agent.ent_coef:.2g}", flush=True)

        # 最优存档: stage 越高权重越大, 同 stage 内取成功率新高
        score = stage * 10.0 + succ_rate
        if stage_ep_count >= args.advance_min_episodes and score > best_succ:
            best_succ = score
            agent.save(os.path.join(out_dir, "best.pt"),
                       extra=dict(iter=it, stage=stage, succ_rate=succ_rate))
        if it % args.save_every == 0 or it == args.iters:
            agent.save(os.path.join(out_dir, "latest.pt"),
                       extra=dict(iter=it, stage=stage, succ_rate=succ_rate))

    agent.save(os.path.join(out_dir, "final.pt"), extra=dict(iter=args.iters, stage=stage))
    print(f"done. 存档于 {out_dir}")


if __name__ == "__main__":
    main()
