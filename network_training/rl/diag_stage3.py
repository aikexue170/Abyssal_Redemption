"""diag_stage3.py —— 逐帧检查 stage 3 下到底是哪个成功条件不满足。"""
import sys

import torch

from ppo import PPO
from tensor_env import STAGES, NavEnv

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/nav_v6/latest.pt"
stage = int(sys.argv[2]) if len(sys.argv) > 2 else 3

ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
agent = PPO(ckpt["obs_dim"], ckpt["act_dim"], device="cuda")
agent.net.load_state_dict(ckpt["state_dict"])

env = NavEnv(num_envs=4, device="cuda", randomize=0.0, seed=7)
env.set_stage(**STAGES[stage])
obs = env.reset()
print(f"stage {stage}: 半径={env.arrival_radius} 停速={env.stop_speed} "
      f"角度={env.success_angle} hold={env.hold_steps}步")
for t in range(env.max_steps):
    a, _, _, _ = agent.net.act(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(a)
    if t % 20 == 0 or t == env.max_steps - 1:
        _, dist, dh = env._dist_dh()
        spd = env.vel.norm(dim=-1)
        i = 0
        ok_zone = bool(dist[i] < env.arrival_radius)
        ok_stop = bool(spd[i] < env.stop_speed)
        ok_alig = bool(abs(dh[i]) < env.success_angle)
        print(f"t={t:3d} dist={dist[i]:7.1f} spd={spd[i]:5.1f} dh={dh[i]:7.1f} "
              f"w={env.ang_vel[i]:6.1f} u=({a[i,0]:5.2f},{a[i,1]:5.2f},{a[i,2]:5.2f}) "
              f"hold={int(env.hold_count[i]):3d} "
              f"zone={ok_zone} stop={ok_stop} align={ok_alig}", flush=True)
print("success_awarded:", env.success_awarded.tolist())
