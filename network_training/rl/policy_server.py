"""policy_server.py —— sim-to-real 部署: 加载导航策略, 经 65432 通道控制游戏里的临渊。

用法 (conda activate starsector, 游戏在战斗中且装了训练桥接器):
    cd network_training/rl
    python policy_server.py --ckpt runs/nav_v7/final.pt

终端命令:
    target <x> <y> [heading_deg]   设置目标位置与目标朝向(朝向省略则保持当前值)
    here                           打印当前船坐标/朝向/速度
    reset                          发送 RESET 回出生姿态
    stop / go                      暂停 / 恢复策略控制
    quit                           退出 (退出前油门归零)

实时窗口:
    蓝点+蓝箭头 = 船当前位置/朝向, 红星+红箭头 = 目标位置/目标朝向,
    灰色尾迹 = 最近 60 秒轨迹。左键点击窗口任意位置 = 设为目标点(朝向不变)。
    左上角文本实时显示坐标读数。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch

from ppo import PPO
from training_server import TrainingServer

STATE_MAX_AGE = 1.0      # 状态超过 1 秒未更新视为断线, 油门归零
CONTROL_HZ = 20          # 控制频率 (状态帧 10Hz, 重复发送无害)


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.ship_id: str | None = None
        self.x = self.y = self.vx = self.vy = 0.0
        self.facing = 0.0
        self.ang_vel = 0.0
        self.sim_time = -1.0
        self.wall_time = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_facing = 0.0
        self.has_target = False
        self.auto = True
        self.quit = False
        self.trail = deque(maxlen=600)   # 最近 60s (10Hz)
        self.act_hist = deque(maxlen=200)  # 最近 ~20s 动作 (10Hz): (t, m, t_, s)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def build_obs(s: SharedState) -> list[float]:
    """与 tensor_env.NavEnv._obs 严格一致的 8 维观测。"""
    relx, rely = s.target_x - s.x, s.target_y - s.y
    dist = math.hypot(relx, rely)
    th = math.radians(s.facing)
    cos, sin = math.cos(th), math.sin(th)
    rel_fwd = cos * relx + sin * rely
    rel_right = -sin * relx + cos * rely
    vf = cos * s.vx + sin * s.vy
    vs = -sin * s.vx + cos * s.vy
    dh = math.radians(wrap_deg(s.target_facing - s.facing))
    return [rel_fwd / 1000.0, rel_right / 1000.0, vf / 60.0, vs / 60.0,
            s.ang_vel / 20.0, math.sin(dh), math.cos(dh), dist / 1000.0]


def control_loop(server: TrainingServer, agent: PPO, s: SharedState):
    last_sent_sim = -1.0
    last_telem = 0.0
    while True:
        with s.lock:
            if s.quit:
                return
            auto, has_target = s.auto, s.has_target
            fresh = (time.time() - s.wall_time) < STATE_MAX_AGE
            new_frame = s.sim_time != last_sent_sim
            if auto and has_target and fresh and new_frame and s.ship_id:
                obs = torch.tensor([build_obs(s)], dtype=torch.float32,
                                   device=agent.device)
                with torch.no_grad():
                    action, _, _, _ = agent.net.act(obs, deterministic=True)
                a = action[0].tolist()
                sid = s.ship_id
            else:
                a = None
                sid = None
        if a is not None:
            server.send_action(sid, move=a[0], turn=a[1], strafe=a[2])
            last_sent_sim = s.sim_time
            with s.lock:
                s.act_hist.append((time.time(), a[0], a[1], a[2]))
        now = time.time()
        if now - last_telem > 2.0:
            last_telem = now
            with s.lock:
                d = math.hypot(s.target_x - s.x, s.target_y - s.y)
                recent = [h for h in s.act_hist if now - h[0] < 2.0]
                if recent:
                    duty = sum(1 for h in recent if abs(h[1]) > 0.5) / len(recent)
                    m = sum(h[1] for h in recent) / len(recent)
                    tn = sum(h[2] for h in recent) / len(recent)
                    sf = sum(h[3] for h in recent) / len(recent)
                    act_str = (f"act(m/t/s)=({m:5.2f},{tn:5.2f},{sf:5.2f}) "
                               f"duty={duty:4.0%} n={len(recent)}")
                else:
                    act_str = "act=--"
                print(f"[telem] pos=({s.x:.0f},{s.y:.0f}) dist={d:.0f} "
                      f"speed={math.hypot(s.vx, s.vy):.1f} "
                      f"dh={wrap_deg(s.target_facing - s.facing):.1f} {act_str}",
                      flush=True)
        time.sleep(1.0 / CONTROL_HZ)


def input_loop(server: TrainingServer, s: SharedState):
    print("命令: target <x> <y> [heading] | here | reset | stop | go | quit")
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:
            return
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        with s.lock:
            if cmd == "target" and len(parts) >= 3:
                s.target_x, s.target_y = float(parts[1]), float(parts[2])
                if len(parts) >= 4:
                    s.target_facing = float(parts[3])
                s.has_target = True
                print(f"[target] ({s.target_x:.0f}, {s.target_y:.0f}) "
                      f"heading {s.target_facing:.0f}°", flush=True)
            elif cmd == "here":
                print(f"[here] pos=({s.x:.1f}, {s.y:.1f}) facing={s.facing:.1f}° "
                      f"speed={math.hypot(s.vx, s.vy):.1f} angvel={s.ang_vel:.1f}",
                      flush=True)
            elif cmd == "reset":
                server.send_reset(s.ship_id or "ALL")
                print("[reset] 已发送", flush=True)
            elif cmd == "stop":
                s.auto = False
                if s.ship_id:
                    server.send_action(s.ship_id, 0.0, 0.0, 0.0)
                print("[stop] 策略暂停, 油门归零", flush=True)
            elif cmd == "go":
                s.auto = True
                print("[go] 策略恢复", flush=True)
            elif cmd == "quit":
                s.quit = True
                return
            else:
                print("未知命令", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__),
                                                   "runs", "nav_v7", "final.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ship", default=None, help="指定 shipId, 默认取第一艘")
    ap.add_argument("--target", nargs="+", type=float, default=None,
                    metavar=("X", "Y"), help="初始目标: x y [heading]")
    ap.add_argument("--relpos", nargs="+", type=float, default=None,
                    metavar=("DX", "DY"), help="相对出生点偏移: dx dy [dheading]")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    agent = PPO(ckpt["obs_dim"], ckpt["act_dim"], device=args.device)
    agent.net.load_state_dict(ckpt["state_dict"])
    print(f"策略已加载: {args.ckpt} (obs={ckpt['obs_dim']}, act={ckpt['act_dim']})")

    s = SharedState()
    server = TrainingServer()
    server.start()

    def on_frame(states):
        with s.lock:
            st = states[0]
            if s.ship_id is None:
                s.ship_id = st.ship_id
                if args.target:
                    s.target_x, s.target_y = args.target[0], args.target[1]
                    s.target_facing = args.target[2] if len(args.target) > 2 else st.facing
                    s.has_target = True
                elif args.relpos:
                    s.target_x, s.target_y = st.x + args.relpos[0], st.y + args.relpos[1]
                    s.target_facing = st.facing + (args.relpos[2] if len(args.relpos) > 2 else 0.0)
                    s.has_target = True
                else:
                    s.target_x, s.target_y = st.x, st.y
                    s.target_facing = st.facing
                    s.has_target = False
            if args.ship is None or st.ship_id == args.ship or s.ship_id == st.ship_id:
                s.x, s.y = st.x, st.y
                s.vx, s.vy = st.vx, st.vy
                s.facing, s.ang_vel = st.facing, st.ang_vel
                s.sim_time = st.sim_time
                s.wall_time = time.time()
                s.trail.append((st.x, st.y))

    server.set_frame_hook(on_frame)
    print("等待游戏连接... (确认战斗已开始且舰船装了训练桥接器)")
    sid = server.wait_for_ship(timeout=120.0)
    if sid is None:
        print("超时: 未等到舰船")
        return
    with s.lock:
        s.ship_id = args.ship or sid
    print(f"已锁定舰船: {s.ship_id}")

    t_ctrl = threading.Thread(target=control_loop, args=(server, agent, s), daemon=True)
    t_in = threading.Thread(target=input_loop, args=(server, s), daemon=True)
    t_ctrl.start()
    t_in.start()

    # ---------------- 实时窗口 (主线程) ----------------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("临渊导航监控")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    trail_line, = ax.plot([], [], "-", lw=0.8, color="gray", alpha=0.6)
    ship_pt, = ax.plot([], [], "o", color="tab:blue", ms=8)
    ship_ar = ax.arrow(0, 0, 0, 0, head_width=12, color="tab:blue")
    tgt_pt, = ax.plot([], [], "*", color="tab:red", ms=16)
    tgt_ar = ax.arrow(0, 0, 0, 0, head_width=12, color="tab:red", alpha=0.7)
    readout = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                      fontsize=9, family="monospace")
    ax.set_xlabel("x (su)")
    ax.set_ylabel("y (su)")

    def on_click(ev):
        if ev.inaxes is ax and ev.button == 1:
            with s.lock:
                s.target_x, s.target_y = float(ev.xdata), float(ev.ydata)
                s.has_target = True
            print(f"[click] target -> ({ev.xdata:.0f}, {ev.ydata:.0f})", flush=True)
    fig.canvas.mpl_connect("button_press_event", on_click)

    ARROW_LEN = 60.0
    plt.show(block=False)          # 不调用 show() TkAgg 窗口不会映射到桌面
    try:
        while True:
            with s.lock:
                if s.quit:
                    break
                x, y, facing = s.x, s.y, s.facing
                vx, vy, av = s.vx, s.vy, s.ang_vel
                tx, ty, tf = s.target_x, s.target_y, s.target_facing
                has_t, auto = s.has_target, s.auto
                trail = list(s.trail)
            if trail:
                xs, ys = zip(*trail)
                trail_line.set_data(xs, ys)
            ship_pt.set_data([x], [y])
            ship_ar.remove()
            ship_ar = ax.arrow(x, y, ARROW_LEN * math.cos(math.radians(facing)),
                               ARROW_LEN * math.sin(math.radians(facing)),
                               head_width=12, color="tab:blue")
            if has_t:
                tgt_pt.set_data([tx], [ty])
                tgt_ar.remove()
                tgt_ar = ax.arrow(tx, ty, ARROW_LEN * math.cos(math.radians(tf)),
                                  ARROW_LEN * math.sin(math.radians(tf)),
                                  head_width=12, color="tab:red", alpha=0.7)
            dist = math.hypot(tx - x, ty - y)
            dh = wrap_deg(tf - facing)
            readout.set_text(
                f"pos=({x:7.1f},{y:7.1f})  facing={facing:6.1f}°\n"
                f"speed={math.hypot(vx, vy):5.1f}  angvel={av:6.1f}°/s\n"
                f"target=({tx:7.1f},{ty:7.1f})  tfacing={tf:6.1f}°\n"
                f"dist={dist:7.1f}  dh={dh:6.1f}°  auto={auto}")
            # 视野: 包住船、目标和轨迹
            allx = [x, tx] + list(xs if trail else [])
            ally = [y, ty] + list(ys if trail else [])
            pad = 150.0
            ax.set_xlim(min(allx) - pad, max(allx) + pad)
            ax.set_ylim(min(ally) - pad, max(ally) + pad)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        with s.lock:
            sid2 = s.ship_id
        if sid2:
            try:
                server.send_action(sid2, 0.0, 0.0, 0.0)
            except Exception:
                pass
        print("已退出 (油门归零)")


if __name__ == "__main__":
    main()
