"""最小二乘拟合游戏内舰船运动参数，并用 matplotlib 可视化。

模型（对每个通道独立拟合）：
    dv/dt = a * u - b * v
    离散化:  dv = (a*u - b*v) * dt
  - a: 单位开度产生的加速度 (su/s^2 或 度/s^2)
  - b: 阻尼系数 (1/s)，松开后 dv/dt = -b*v 即为减速特性
  - 隐含稳态速度 = a / b；游戏内另有硬速度上限，单独报告观测上限

关键点（历史踩坑）：游戏直接上报的速度与位置存在时序不同步，
位置差分得到的速度更准 —— 因此拟合一律使用位置/角度差分速度，
直接上报速度只用于对比图。

用法:  python analyze_fit.py [samples.csv]
       不传参数则使用 output/ 下最新的 samples_*.csv
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CAP_PERCENTILE = 99       # 观测速度上限取该分位数
CAP_EXCLUDE = 0.9         # 拟合时剔除 |v| > 0.9*cap 的样本（避免硬上限区污染）

CHANNEL_TITLES = {
    "move": "Forward/Backward (u>0 forward)",
    "turn": "Turn (u>0 increases facing)",
    "strafe": "Strafe (u>0 right)",
}


# ---------------- 数据加载 ----------------

def load_samples(path: str):
    """返回 {(channel, trial): {列名: np.array}}"""
    groups = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        import csv
        for row in csv.DictReader(f):
            key = (row["channel"], int(row["trial"]))
            for col, val in row.items():
                if col in ("channel",):
                    continue
                groups[key][col].append(float(val))
    return {k: {c: np.asarray(v) for c, v in cols.items()} for k, cols in groups.items()}


# ---------------- 差分速度 ----------------

def diff_velocity(channel: str, d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (t_mid, dt, v_meas, v_direct, u)：
      v_meas   —— 位置/角度差分速度（拟合用）
      v_direct —— 游戏直接上报速度投影（仅对比用）
      u        —— 该区间起始行施加的开度
    """
    t = d["t"]; x = d["x"]; y = d["y"]; facing = d["facing"]
    dt = np.diff(t)
    dx = np.diff(x); dy = np.diff(y)
    theta = np.radians(facing[:-1])

    if channel == "move":
        v_meas = (dx * np.cos(theta) + dy * np.sin(theta)) / dt
        v_direct = d["vx"][:-1] * np.cos(theta) + d["vy"][:-1] * np.sin(theta)
    elif channel == "strafe":
        s = theta + math.pi / 2
        v_meas = (dx * np.cos(s) + dy * np.sin(s)) / dt
        v_direct = d["vx"][:-1] * np.cos(s) + d["vy"][:-1] * np.sin(s)
    elif channel == "turn":
        df = (np.diff(facing) + 180.0) % 360.0 - 180.0   # 角度解卷绕
        v_meas = df / dt
        v_direct = d["ang_vel"][:-1]
    else:
        raise ValueError(channel)

    u = d["u_applied"][:-1]
    t_mid = t[:-1]
    return t_mid, dt, v_meas, v_direct, u


# ---------------- 拟合 ----------------

def fit_channel(channel: str, trials: list[dict]):
    """对所有 trial 合并做最小二乘；返回参数字典与逐 trial 的差分序列。"""
    seqs = []
    for d in trials:
        t_mid, dt, v_meas, v_direct, u = diff_velocity(channel, d)
        ok = dt > 1e-4
        seqs.append(dict(t=t_mid[ok], dt=dt[ok], v=v_meas[ok],
                         vd=v_direct[ok], u=u[ok], raw=d))

    v_active = np.concatenate([s["v"][s["u"] != 0] for s in seqs if np.any(s["u"] != 0)])
    if v_active.size == 0:
        raise RuntimeError(f"通道 {channel} 没有动作生效期间的样本")
    cap = np.percentile(np.abs(v_active), CAP_PERCENTILE)

    # v 是逐区间差分速度序列；dv_k = v_{k+1} - v_k，回归量取区间 k 的 u, dt, v。
    # 剔除 |v| 接近硬上限的样本（上限区不再线性），以及复位传送造成的异常跳变。
    rows_x, rows_y = [], []
    for s in seqs:
        dv = np.diff(s["v"])
        u_k = s["u"][:-1]; dt_k = s["dt"][:-1]; v_k = s["v"][:-1]
        mask = (np.abs(v_k) < CAP_EXCLUDE * cap) & (np.abs(dv) < 10 * cap)
        rows_x.append(np.stack([u_k[mask] * dt_k[mask], -v_k[mask] * dt_k[mask]], axis=1))
        rows_y.append(dv[mask])

    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    (a, b), *_ = np.linalg.lstsq(X, Y, rcond=None)
    y_hat = X @ np.array([a, b])
    ss_res = float(np.sum((Y - y_hat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "accel_per_unit": float(a),          # 单位开度加速度
        "damping": float(b),                 # 阻尼系数 1/s
        "implied_max_speed": float(a / b) if b > 1e-9 else None,
        "observed_speed_cap": float(cap),    # 观测速度上限（硬上限附近）
        "r2": r2,
        "n_samples": int(X.shape[0]),
    }, seqs


def simulate(seq: dict, a: float, b: float) -> np.ndarray:
    """用拟合参数对单条 trial 前向仿真，用于和实测对比。"""
    v = np.zeros_like(seq["v"])
    v[0] = seq["v"][0]
    for k in range(len(v) - 1):
        v[k + 1] = v[k] + (a * seq["u"][k] - b * v[k]) * seq["dt"][k]
    return v


# ---------------- 绘图 ----------------

def plot_channel(channel: str, params: dict, seqs: list[dict], outdir: str) -> str:
    by_u = defaultdict(list)
    for s in seqs:
        nonzero = s["u"][np.nonzero(s["u"])]
        key = round(float(nonzero[0]), 2) if nonzero.size else 0.0
        by_u[key].append(s)

    fig, axes = plt.subplots(1, len(by_u), figsize=(6 * len(by_u), 4.5), sharey=True)
    if len(by_u) == 1:
        axes = [axes]
    for ax, (u_val, group) in zip(axes, sorted(by_u.items(), reverse=True)):
        for s in group:
            nz = np.nonzero(s["u"])[0]
            t0 = s["t"][nz[0]] if nz.size else s["t"][0]
            ax.plot(s["t"] - t0, s["v"], alpha=0.45, lw=1,
                    color="tab:blue", label="_measured")
            ax.plot(s["t"] - t0, simulate(s, params["accel_per_unit"], params["damping"]),
                    ls="--", lw=1.6, color="tab:red", label="_model")
        ax.axhline(params["observed_speed_cap"] * np.sign(u_val), color="gray", ls=":", lw=1)
        ax.set_title(f"u = {u_val:+.2f}")
        ax.set_xlabel("t since action start (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("velocity (su/s or deg/s)")
    handles = [plt.Line2D([], [], color="tab:blue", label="measured (pos-diff)"),
               plt.Line2D([], [], color="tab:red", ls="--", label="fitted model"),
               plt.Line2D([], [], color="gray", ls=":", label="observed cap")]
    fig.legend(handles=handles, loc="upper right")
    vmax = params['implied_max_speed']
    vmax_str = f"{vmax:.1f}" if vmax is not None else "inf"
    fig.suptitle(f"{CHANNEL_TITLES.get(channel, channel)}\n"
                 f"a={params['accel_per_unit']:.2f}, b={params['damping']:.3f}, "
                 f"v_max(implied)={vmax_str}, "
                 f"cap={params['observed_speed_cap']:.1f}, R2={params['r2']:.3f}")
    fig.tight_layout()
    path = os.path.join(outdir, f"fit_{channel}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_source_compare(seqs: list[dict], outdir: str) -> str:
    """位置差分速度 vs 游戏直接上报速度（展示时序不同步问题）。"""
    s = max(seqs, key=lambda s: float(np.max(np.abs(s["v"]))))
    nz = np.nonzero(s["u"])[0]
    t0 = s["t"][nz[0]] if nz.size else s["t"][0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s["t"] - t0, s["v"], lw=1.8, label="position-diff velocity (used for fitting)")
    ax.plot(s["t"] - t0, s["vd"], lw=1.2, alpha=0.8, label="game-reported velocity (desynced)")
    ax.set_xlabel("t since action start (s)")
    ax.set_ylabel("velocity")
    ax.set_title("Why position differencing: game velocity is time-desynced")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "velocity_source_compare.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------- 主流程 ----------------

def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "samples_*.csv")))
        if not files:
            print("未找到样本文件，请先运行 run_sampling.py")
            return
        path = files[-1]
    print(f"[拟合] 样本文件: {path}")

    groups = load_samples(path)
    channels = sorted({c for c, _ in groups})
    params_all: dict[str, dict] = {}
    seqs_by_channel: dict[str, list] = {}

    for channel in channels:
        trials = [groups[k] for k in sorted(groups) if k[0] == channel]
        params, seqs = fit_channel(channel, trials)
        params_all[channel] = params
        seqs_by_channel[channel] = seqs
        print(f"[拟合] {channel:7s}  a={params['accel_per_unit']:9.3f}  "
              f"b={params['damping']:7.4f}  v_max={params['implied_max_speed']}  "
              f"cap={params['observed_speed_cap']:8.2f}  R2={params['r2']:.4f}  "
              f"n={params['n_samples']}")
        p = plot_channel(channel, params, seqs, OUTPUT_DIR)
        print(f"        图 -> {p}")

    # 用 move 通道做一次差分 vs 直报速度对比图
    if "move" in seqs_by_channel:
        p = plot_source_compare(seqs_by_channel["move"], OUTPUT_DIR)
        print(f"        图 -> {p}")

    params_all["_meta"] = {
        "model": "dv = (a*u - b*v) * dt",
        "source_file": os.path.basename(path),
        "note": "速度/角速度均由位置与朝向差分得到；游戏直接上报速度存在时序不同步，未用于拟合。",
    }
    out = os.path.join(OUTPUT_DIR, "params.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(params_all, f, ensure_ascii=False, indent=2)
    print(f"[拟合] 参数 -> {out}")


if __name__ == "__main__":
    main()
