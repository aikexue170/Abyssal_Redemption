"""最小二乘拟合游戏内舰船运动参数，并用 matplotlib 可视化。

模型（每个通道独立，一切从简）：
    油门期:  dv/dt = a * u              —— 恒定加速度，与 v 无关（相图呈水平条带）
            越过施力上限 force_limit 后只剩很小的恒定爬行 creep（游戏引擎残余效应）
    上限:    v 最终钳制在 [v_cap_neg, v_cap_pos]
    滑行期:  dv/dt ≈ 0                  —— 阻力近似为零（实测极小，见 params.json 备注）

关键点（历史踩坑）：游戏直接上报的速度与位置存在时序不同步，
位置差分得到的速度更准 —— 拟合一律使用位置/角度差分速度。

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
CAP_PERCENTILE = 99       # 速度上限观测取该分位数
FIT_REGION = 0.9          # 拟合 a 只用 |v| < 0.9*cap 的样本（避开上限截断区）

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

def diff_velocity(channel: str, d: dict):
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

    return t[:-1], dt, v_meas, v_direct, d["u_applied"][:-1]


# ---------------- 拟合 ----------------

def fit_channel(channel: str, trials: list[dict]):
    """最小二乘拟合 a（油门期 dv/dt = a*u）与双向速度上限。"""
    seqs = []
    for d in trials:
        t_mid, dt, v, vd, u = diff_velocity(channel, d)
        ok = dt > 1e-4
        seqs.append(dict(t=t_mid[ok], dt=dt[ok], v=v[ok],
                         vd=vd[ok], u=u[ok], raw=d))

    # 1) 双向速度上限：油门期间 |v| 的高分位数
    v_pos = np.concatenate([s["v"][s["u"] > 0] for s in seqs if np.any(s["u"] > 0)])
    v_neg = np.concatenate([s["v"][s["u"] < 0] for s in seqs if np.any(s["u"] < 0)])
    cap_pos = float(np.percentile(v_pos, CAP_PERCENTILE)) if v_pos.size else 0.0
    cap_neg = float(np.percentile(v_neg, 100 - CAP_PERCENTILE)) if v_neg.size else 0.0

    # 2) 分方向拟合。施力区 dv/dt = a*u；越过施力上限后只剩很小的恒定爬行，
    #    爬行段样本数量多、会淹没回归，因此每个 trial 只取加速段前 5%~40%
    #    平台速度的"干净斜坡窗口"，窗口内取 dv/dt 中位数，再对 u 过原点回归。
    def collect(s):
        dv = np.diff(s["v"])
        u_k = s["u"][:-1]; dt_k = s["dt"][:-1]; v_k = s["v"][:-1]
        jump = np.abs(dv) < 10 * max(cap_pos, -cap_neg)   # 剔除复位传送跳变
        return u_k, v_k, dv / dt_k, jump

    def fit_direction(sign):
        us, dvdts = [], []
        for s in seqs:
            u_k, v_k, dvdt, jump = collect(s)
            act = (u_k * sign > 0) & jump
            if not np.any(act):
                continue
            plateau = np.max(np.abs(s["v"][np.r_[False, act]]))
            lo, hi = 0.05 * plateau, 0.4 * plateau
            ramp = act & (np.abs(v_k) >= lo) & (np.abs(v_k) <= hi)
            if np.sum(ramp) < 3:
                ramp = act                            # 窗口太窄则退化为全部油门样本
            # 每个 trial 取中位数，抗爬行段/异常点污染
            us.append(float(np.median(u_k[act])))
            dvdts.append(float(np.median(dvdt[ramp])))
        U = np.array(us); D = np.array(dvdts)
        a = float(U @ D / (U @ U))
        # 施力上限：该方向油门样本中 |dv/dt| 仍高于 a*|u|/2 的最大 |v|
        lim = 0.0
        for s in seqs:
            u_k, v_k, dvdt, jump = collect(s)
            mask = (u_k * sign > 0) & jump & (dvdt * sign > 0.5 * a * np.abs(u_k))
            if np.any(mask):
                lim = max(lim, float(np.max(np.abs(v_k[mask]))))
        # 爬行：施力上限外油门样本的平均 dv/dt
        cr = []
        for s in seqs:
            u_k, v_k, dvdt, jump = collect(s)
            beyond = (v_k > lim) if sign > 0 else (v_k < -lim)
            mask = (u_k * sign > 0) & beyond & jump
            if np.any(mask):
                cr.append(dvdt[mask])
        return a, lim, (float(np.mean(np.concatenate(cr))) if cr else 0.0)

    a_pos, lim_pos, creep_pos = fit_direction(+1)
    a_neg, lim_neg, creep_neg = fit_direction(-1)

    # R²：在全部施力区样本上全局评估 dv/dt 预测
    dvs, preds = [], []
    for s in seqs:
        u_k, v_k, dvdt, jump = collect(s)
        pred = np.where(u_k > 0, a_pos * u_k, np.where(u_k < 0, a_neg * u_k, 0.0))
        forcing = ((u_k > 0) & (v_k < lim_pos)) | ((u_k < 0) & (v_k > -lim_neg))
        mask = forcing & jump
        if np.any(mask):
            dvs.append(dvdt[mask]); preds.append(pred[mask])
    D = np.concatenate(dvs); P = np.concatenate(preds)
    r2 = 1.0 - float(((D - P) ** 2).sum()) / float(((D - D.mean()) ** 2).sum())

    # 3) 滑行期实测减速度（仅供参考，模型中按 0 处理）
    coast = []
    for s in seqs:
        dv = np.diff(s["v"])
        u_k = s["u"][:-1]; dt_k = s["dt"][:-1]; v_k = s["v"][:-1]
        mask = (u_k == 0) & (np.abs(v_k) > 1.0)
        if np.any(mask):
            coast.append((-dv[mask] / dt_k[mask]) * np.sign(v_k[mask]))
    coast_decel = float(np.mean(np.concatenate(coast))) if coast else 0.0

    return {
        "accel_per_unit_pos": a_pos,   # 正向单位开度加速度 (su/s^2 或 度/s^2)
        "accel_per_unit_neg": a_neg,   # 负向单位开度加速度（作用于 u<0）
        "force_limit_pos": lim_pos,    # 正向施力上限（超过后进入爬行区）
        "force_limit_neg": lim_neg,    # 负向施力上限（正数，越过 -lim_neg 后爬行）
        "creep_pos": creep_pos,        # 正向爬行加速度（常数）
        "creep_neg": creep_neg,        # 负向爬行加速度（常数，通常为负）
        "v_cap_pos": cap_pos,          # 正向速度最终上限（钳制）
        "v_cap_neg": cap_neg,          # 负向速度最终上限（钳制，负值）
        "coast_decel_measured": coast_decel,  # 实测滑行减速度（模型按 0 处理）
        "r2": r2,
    }, seqs


def simulate(seq: dict, params: dict) -> np.ndarray:
    """前向仿真：施力区 dv = a*u*dt；越过施力上限 dv = creep*dt；
    最终钳制在 [v_cap_neg, v_cap_pos]；滑行无阻力。"""
    v = np.zeros_like(seq["v"])
    v[0] = np.clip(seq["v"][0], params["v_cap_neg"], params["v_cap_pos"])
    for k in range(len(v) - 1):
        u = seq["u"][k]
        if u > 0:
            dv = (params["accel_per_unit_pos"] * u
                  if v[k] < params["force_limit_pos"] else params["creep_pos"]) * seq["dt"][k]
        elif u < 0:
            dv = (params["accel_per_unit_neg"] * u
                  if v[k] > -params["force_limit_neg"] else params["creep_neg"]) * seq["dt"][k]
        else:
            dv = 0.0
        v[k + 1] = np.clip(v[k] + dv, params["v_cap_neg"], params["v_cap_pos"])
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
            ax.plot(s["t"] - t0, simulate(s, params),
                    ls="--", lw=1.6, color="tab:red", label="_model")
        cap_line = params["v_cap_pos"] if u_val > 0 else params["v_cap_neg"]
        ax.axhline(cap_line, color="gray", ls=":", lw=1)
        ax.set_title(f"u = {u_val:+.2f}")
        ax.set_xlabel("t since action start (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("velocity (su/s or deg/s)")
    handles = [plt.Line2D([], [], color="tab:blue", label="measured (pos-diff)"),
               plt.Line2D([], [], color="tab:red", ls="--", label="fitted model"),
               plt.Line2D([], [], color="gray", ls=":", label="speed cap")]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"{CHANNEL_TITLES.get(channel, channel)}\n"
                 f"a+={params['accel_per_unit_pos']:.2f}, a-={params['accel_per_unit_neg']:.2f}, "
                 f"force_limit=[{-params['force_limit_neg']:.1f}, {params['force_limit_pos']:.1f}], "
                 f"cap=[{params['v_cap_neg']:.1f}, {params['v_cap_pos']:.1f}], "
                 f"R2={params['r2']:.3f}")
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
        print(f"[拟合] {channel:7s}  a+={params['accel_per_unit_pos']:9.3f}  "
              f"a-={params['accel_per_unit_neg']:9.3f}  "
              f"force_limit=[{-params['force_limit_neg']:7.2f}, {params['force_limit_pos']:7.2f}]  "
              f"cap=[{params['v_cap_neg']:7.2f}, {params['v_cap_pos']:7.2f}]  "
              f"R2={params['r2']:.4f}")
        p = plot_channel(channel, params, seqs, OUTPUT_DIR)
        print(f"        图 -> {p}")

    if "move" in seqs_by_channel:
        p = plot_source_compare(seqs_by_channel["move"], OUTPUT_DIR)
        print(f"        图 -> {p}")

    params_all["_meta"] = {
        "model": "dv = a*u*dt within force_limit, creep beyond, clamped to [v_cap_neg, v_cap_pos]; coast drag approximated as 0",
        "source_file": os.path.basename(path),
        "note": "速度/角速度由位置与朝向差分得到；游戏直报速度存在时序不同步，未用于拟合。"
                "coast_decel_measured 为实测滑行减速度（很小），模型中按 0 处理。",
    }
    out = os.path.join(OUTPUT_DIR, "params.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(params_all, f, ensure_ascii=False, indent=2)
    print(f"[拟合] 参数 -> {out}")


if __name__ == "__main__":
    main()
