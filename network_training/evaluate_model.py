"""模型评估：真实游戏轨迹 vs 拟合环境仿真轨迹。

流程：
    1. 连接游戏，复位后按 MOTION_SCRIPT 播放一段复合运动指令（约 30 秒），
       全程记录真实状态（output/eval_*.csv）
    2. 用 output/params.json 的拟合参数，以 dt=0.05 离线仿真同一指令序列
       （初始姿态取真实记录的第一帧）
    3. 两条 x-y 轨迹画在同一张图上，并给出位置误差曲线与统计指标

用法:  python evaluate_model.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training_server import TrainingServer
from run_sampling import Recorder, sim_sleep, wait_settled

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ---------------- 复合运动脚本 ----------------
# (持续秒数[游戏内], move, turn, strafe)，总计约 30 秒
MOTION_SCRIPT: list[tuple[float, float, float, float]] = [
    (0.5,  0.0,  0.0,  0.0),   # 静止基线
    (5.0,  1.0,  0.0,  0.0),   # 全速前进
    (4.0,  0.0,  1.0,  0.0),   # 原地转向
    (5.0,  0.5,  0.5,  0.0),   # 前进+转向（弧线）
    (4.0,  0.0,  0.0,  1.0),   # 向右平移
    (4.0, -0.5, -0.5,  0.0),   # 倒退+回转
    (3.5,  0.8,  0.0, -0.5),   # 前进+向左平移
    (4.0,  0.0,  0.0,  0.0),   # 松开滑行
]

SIM_DT = 0.05                 # 离线仿真步长（秒）
EVAL_WINDOW = 30.0            # 误差统计窗口（秒）
TARGET_SHIP_ID = None         # None = 自动选第一艘


# ---------------- 拟合模型（与 analyze_fit.simulate 同一套分段规则） ----------------

def channel_step(v: float, u: float, p: dict, dt: float) -> float:
    if u > 0:
        dv = (p["accel_per_unit_pos"] * u if v < p["force_limit_pos"] else p["creep_pos"]) * dt
    elif u < 0:
        dv = (p["accel_per_unit_neg"] * u if v > -p["force_limit_neg"] else p["creep_neg"]) * dt
    else:
        dv = 0.0
    return float(np.clip(v + dv, p["v_cap_neg"], p["v_cap_pos"]))


def channel_accel_world(v_proj: float, u: float, p: dict) -> float:
    """世界系模型的通道加速度：施力与否取决于速度在船体系上的投影
    （与 ShipControlSystem 的投影限速逻辑一致）。返回沿通道方向的标量加速度。"""
    if u > 0:
        if v_proj < p["force_limit_pos"]:
            return p["accel_per_unit_pos"] * u
        return p["creep_pos"] if v_proj < p["v_cap_pos"] else 0.0
    if u < 0:
        if v_proj > -p["force_limit_neg"]:
            return p["accel_per_unit_neg"] * u
        return p["creep_neg"] if v_proj > p["v_cap_neg"] else 0.0
    return 0.0


def simulate(rec: list[dict], params: dict, dt: float, coast_drag: bool = False):
    """世界系惯性模型：速度不随船体旋转，通道力沿船体轴施加；
    转向为船体系角速度通道。coast_drag=True 时对松开油门的通道施加
    params 里实测的恒定减速度（库仑式，仅 turn 通道明显）。"""
    t0 = rec[0]["t"]
    x, y, facing = rec[0]["x"], rec[0]["y"], rec[0]["facing"]
    vx = vy = w = 0.0          # 复位后静止出发

    rec_t = np.array([r["t"] for r in rec])
    rec_move = np.array([r["u_move"] for r in rec])
    rec_turn = np.array([r["u_turn"] for r in rec])
    rec_strafe = np.array([r["u_strafe"] for r in rec])

    t_end = rec_t[-1]
    n = int((t_end - t0) / dt) + 1
    ts = t0 + np.arange(n) * dt
    idx = np.searchsorted(rec_t, ts, side="right") - 1   # 阶梯插值
    idx = np.clip(idx, 0, len(rec) - 1)

    pm, pt, ps = params["move"], params["turn"], params["strafe"]
    out = []
    for k in range(n):
        out.append((ts[k], x, y, facing, vx, vy, w))
        um, ut, us_ = rec_move[idx[k]], rec_turn[idx[k]], rec_strafe[idx[k]]

        th = math.radians(facing)
        fx, fy = math.cos(th), math.sin(th)      # 船头方向
        rx, ry = -math.sin(th), math.cos(th)     # 右舷方向

        vf = vx * fx + vy * fy
        vs = vx * rx + vy * ry

        # move / strafe：通道力沿船体轴施加到世界速度上
        a = channel_accel_world(vf, um, pm)
        if a == 0.0 and um == 0.0 and coast_drag:
            a = -pm["coast_decel_measured"] * math.copysign(1.0, vf) if abs(vf) > 1e-6 else 0.0
        vx += a * fx * dt; vy += a * fy * dt

        a = channel_accel_world(vs, us_, ps)
        if a == 0.0 and us_ == 0.0 and coast_drag:
            a = -ps["coast_decel_measured"] * math.copysign(1.0, vs) if abs(vs) > 1e-6 else 0.0
        vx += a * rx * dt; vy += a * ry * dt

        # turn：船体系角速度通道
        w = channel_step(w, ut, pt, dt)
        if ut == 0.0 and coast_drag and abs(w) > 1e-6:
            w -= pt["coast_decel_measured"] * math.copysign(1.0, w) * dt

        facing += w * dt
        x += vx * dt; y += vy * dt
    return np.array(out)   # 列: t, x, y, facing, vx, vy, w


# ---------------- 记录（扩展 Recorder：每帧带全三轴指令） ----------------

class EvalRecorder(Recorder):
    COLUMNS = ["trial", "channel", "u_applied", "u_move", "u_turn", "u_strafe",
               "t", "x", "y", "vx", "vy", "facing", "ang_vel"]

    def __init__(self, ship_id_getter):
        super().__init__(ship_id_getter)
        self.u3 = (0.0, 0.0, 0.0)

    def set_applied3(self, move: float, turn: float, strafe: float) -> None:
        with self._lock:
            self.u3 = (move, turn, strafe)

    def hook(self, states) -> None:
        ship_id = self._ship_id_getter()
        with self._lock:
            if not self.recording or ship_id is None:
                return
            um, ut, us = self.u3
            for s in states:
                if s.ship_id != ship_id:
                    continue
                self.rows.append({
                    "trial": 1, "channel": "eval", "u_applied": um,
                    "u_move": um, "u_turn": ut, "u_strafe": us,
                    "t": s.sim_time, "x": s.x, "y": s.y,
                    "vx": s.vx, "vy": s.vy,
                    "facing": s.facing, "ang_vel": s.ang_vel,
                })


# ---------------- 主流程 ----------------

def main() -> None:
    params_path = os.path.join(OUTPUT_DIR, "params.json")
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    print(f"[评估] 参数 <- {params_path}")

    server = TrainingServer()
    server.start()
    server.wait_connected(timeout=None)

    ship_id = TARGET_SHIP_ID or server.wait_for_ship(timeout=120)
    if ship_id is None:
        print("[评估] 未等到舰船状态，退出")
        return
    print(f"[评估] 目标舰船: {ship_id}")

    recorder = EvalRecorder(lambda: ship_id)
    server.set_frame_hook(recorder.hook)

    try:
        # 复位并静止
        server.send_action(ship_id, 0.0, 0.0, 0.0)
        server.send_reset(ship_id)
        wait_settled(server, ship_id)

        # 播放运动脚本
        recorder.begin_trial(1, "eval")
        for dur, m, tr, st in MOTION_SCRIPT:
            server.send_action(ship_id, m, tr, st)
            recorder.set_applied3(m, tr, st)
            print(f"[评估] {dur:4.1f}s  move={m:+.1f} turn={tr:+.1f} strafe={st:+.1f}")
            sim_sleep(server, ship_id, dur, real_timeout=180.0)
        server.send_action(ship_id, 0.0, 0.0, 0.0)
        recorder.end_trial()
    except KeyboardInterrupt:
        print("\n[评估] 用户中断")
    finally:
        server.send_action(ship_id, 0.0, 0.0, 0.0)
        server.set_frame_hook(None)

    if not recorder.rows:
        print("[评估] 没有记录到数据")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"eval_{stamp}.csv")
    recorder.save(csv_path)

    rec = list(recorder.rows)
    t0 = rec[0]["t"]

    # 离线仿真（与记录等长）：无阻力 / 带实测恒定滑行阻力 两个版本
    sim = simulate(rec, params, SIM_DT, coast_drag=False)
    sim_d = simulate(rec, params, SIM_DT, coast_drag=True)

    # 真实轨迹重采样到仿真时间轴（最近邻），用于逐点误差
    rec_t = np.array([r["t"] for r in rec])
    rec_x = np.array([r["x"] for r in rec])
    rec_y = np.array([r["y"] for r in rec])

    def err_of(s):
        idx = np.clip(np.searchsorted(rec_t, s[:, 0], side="right") - 1, 0, len(rec) - 1)
        return np.hypot(s[:, 1] - rec_x[idx], s[:, 2] - rec_y[idx])

    err = err_of(sim)
    err_d = err_of(sim_d)

    win = sim[:, 0] - t0 <= EVAL_WINDOW
    path_len = float(np.sum(np.hypot(np.diff(rec_x), np.diff(rec_y))))
    print(f"[评估] 窗口 {EVAL_WINDOW}s | 真实轨迹长 {path_len:.0f} su")
    print(f"[评估] 无阻力模型  位置误差: 平均 {err[win].mean():.2f} su, "
          f"最大 {err[win].max():.2f} su, 末端 {err[win][-1]:.2f} su")
    print(f"[评估] 带阻力模型  位置误差: 平均 {err_d[win].mean():.2f} su, "
          f"最大 {err_d[win].max():.2f} su, 末端 {err_d[win][-1]:.2f} su")

    # 绘图：左 x-y 轨迹，右 误差曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(rec_x, rec_y, lw=1.8, color="tab:blue", label="game (measured)")
    ax1.plot(sim[:, 1], sim[:, 2], lw=1.4, ls="--", color="tab:red", label="model, no drag")
    ax1.plot(sim_d[:, 1], sim_d[:, 2], lw=1.2, ls=":", color="tab:green", label="model, with coast drag")
    ax1.plot(rec_x[0], rec_y[0], "ko", ms=6)
    ax1.set_title(f"Trajectory, {sim[-1,0]-t0:.1f}s compound motion")
    ax1.set_xlabel("x (su)"); ax1.set_ylabel("y (su)")
    ax1.axis("equal"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(sim[win, 0] - t0, err[win], lw=1.4, color="tab:red",
             label=f"no drag (mean {err[win].mean():.1f} su)")
    ax2.plot(sim_d[win, 0] - t0, err_d[win], lw=1.4, color="tab:green",
             label=f"with coast drag (mean {err_d[win].mean():.1f} su)")
    ax2.set_title("Position error vs time")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("error (su)")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"evaluation_{stamp}.png")
    fig.savefig(png_path, dpi=140)
    print(f"[评估] 图 -> {png_path}")


if __name__ == "__main__":
    main()
