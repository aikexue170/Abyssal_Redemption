"""环境参数采样：向游戏内舰船施加一组标准动作并记录状态时间序列。

用法：
    1. 构建 jar，启动游戏，进入战斗，给被测舰船装上"训练桥接器"插件
    2. python run_sampling.py
    3. 采样完成后运行 analyze_fit.py 进行最小二乘拟合与可视化

流程：每个动作重复 REPEATS 次；每次先 RESET 归位 -> 静止基线 ->
      施加动作直到平台期（顶到速度上限或稳态）-> 松开滑行直到速度衰减。
      全部样本写入 output/samples_*.csv。
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from datetime import datetime

from training_server import TrainingServer
from protocol import ShipState

# ---------------- 采样配置 ----------------

# 待测动作矩阵：(通道, 开度)。通道取值 move/turn/strafe，对应 ShipControlSystem 三轴。
# 注意：必须包含小开度动作 —— 满开度下速度迅速顶到硬上限，线性区样本太少，
#       小开度（隐含稳态速度低于上限）的完整轨迹才是最小二乘拟合的主要依据。
ACTIONS: list[tuple[str, float]] = [
    ("move", +1.0),
    ("move", -1.0),
    ("move", +0.5),
    ("move", -0.5),
    ("move", +0.2),
    ("move", -0.2),
    ("turn", +1.0),
    ("turn", -1.0),
    ("turn", +0.2),
    ("turn", -0.2),
    ("strafe", +1.0),
    ("strafe", -1.0),
    ("strafe", +0.2),
    ("strafe", -0.2),
]
REPEATS = 3                # 每组动作重复次数

# --drag 模式：阻力律测量。加速到上限后松开，一直记录到完全停住，
# 用于刻画分段阻力曲线的完整形状
DRAG_ACTIONS: list[tuple[str, float]] = [
    ("move", +1.0),
    ("move", -1.0),
    ("turn", +1.0),
    ("turn", -1.0),
    ("strafe", +1.0),
    ("strafe", -1.0),
]
DRAG_REPEATS = 2
COAST_FULL_TIMEOUT = 150.0  # 全程滑行最长观测（游戏内秒）
COAST_FULL_STOP_EPS = 0.2   # 低于该速度视为停住（su/s 或 度/s）
BASELINE_TIME = 0.5        # 施加动作前的静止基线（游戏内秒）

# 动作持续时间是自适应的：持续到检测到平台期（速度不再变化）为止
ACT_TIME_MIN = 4.0         # 动作最短持续（游戏内秒）
ACT_TIME_MAX = 25.0        # 动作最长持续（游戏内秒），保证能顶到速度上限
PLATEAU_WINDOW = 1.5       # 平台期判定的观测窗口（游戏内秒）
PLATEAU_REL_EPS = 0.03     # 窗口内波动小于峰值的 3% 视为进入平台期

# 滑行观测也是自适应的：持续到速度衰减到峰值的 10% 以下为止
COAST_TIME_MIN = 2.0       # 滑行最短观测（游戏内秒）
COAST_TIME_MAX = 8.0       # 滑行最长观测（游戏内秒）
COAST_STOP_RATIO = 0.1     # 速度衰减到峰值该比例以下即停止
COAST_STOP_FLOOR = 0.3     # 或衰减到该绝对值以下（su/s 或 度/s）

SETTLE_SPEED_EPS = 1.0     # 复位后判定静止的速度阈值（su/s）
SETTLE_ANGVEL_EPS = 2.0    # 复位后判定静止的角速度阈值（度/s）
SETTLE_TIMEOUT = 20.0      # 等待静止的真实时间上限（秒）

TARGET_SHIP_ID = None      # None = 自动选择第一艘上报状态的舰船

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ---------------- 记录器 ----------------

class Recorder:
    """把每帧状态追加到内存表；trial/channel/u 由主线程在采样时设置。"""

    COLUMNS = ["trial", "channel", "u_applied", "t", "x", "y",
               "vx", "vy", "facing", "ang_vel"]

    def __init__(self, ship_id_getter):
        self._lock = threading.Lock()
        self.rows: list[dict] = []
        self.trial = -1
        self.channel = ""
        self.u_applied = 0.0
        self.recording = False
        self._ship_id_getter = ship_id_getter

    def begin_trial(self, trial: int, channel: str) -> None:
        with self._lock:
            self.trial = trial
            self.channel = channel
            self.u_applied = 0.0
            self.recording = True

    def set_applied(self, u: float) -> None:
        with self._lock:
            self.u_applied = u

    def end_trial(self) -> None:
        with self._lock:
            self.recording = False

    def hook(self, states: list[ShipState]) -> None:
        ship_id = self._ship_id_getter()
        with self._lock:
            if not self.recording or ship_id is None:
                return
            for s in states:
                if s.ship_id != ship_id:
                    continue
                self.rows.append({
                    "trial": self.trial,
                    "channel": self.channel,
                    "u_applied": self.u_applied,
                    "t": s.sim_time,
                    "x": s.x, "y": s.y,
                    "vx": s.vx, "vy": s.vy,
                    "facing": s.facing, "ang_vel": s.ang_vel,
                })

    def save(self, path: str) -> None:
        with self._lock:
            rows = list(self.rows)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Recorder] 已保存 {len(rows)} 行 -> {path}")


# ---------------- 工具 ----------------

def sim_now(server: TrainingServer, ship_id: str) -> float | None:
    s = server.latest_states().get(ship_id)
    return s.sim_time if s else None


def sim_sleep(server: TrainingServer, ship_id: str, seconds: float, real_timeout: float = 60.0) -> None:
    """等待游戏内时钟前进 seconds 秒（游戏暂停/减速时自动跟随）。"""
    start = sim_now(server, ship_id)
    if start is None:
        time.sleep(seconds)
        return
    deadline = time.time() + real_timeout
    while time.time() < deadline:
        now = sim_now(server, ship_id)
        if now is not None and now - start >= seconds:
            return
        time.sleep(0.02)


def wait_settled(server: TrainingServer, ship_id: str) -> bool:
    """复位后等待舰船静止；超时未静止则重发 RESET。"""
    deadline = time.time() + SETTLE_TIMEOUT
    last_reset = 0.0
    while time.time() < deadline:
        s = server.latest_states().get(ship_id)
        if s and s.speed < SETTLE_SPEED_EPS and abs(s.ang_vel) < SETTLE_ANGVEL_EPS:
            return True
        if time.time() - last_reset > 2.0:
            server.send_reset(ship_id)
            last_reset = time.time()
        time.sleep(0.1)
    print("[采样] 警告：等待静止超时，继续采样")
    return False


def channel_metric(server: TrainingServer, ship_id: str, channel: str) -> float | None:
    """平台期/衰减判定用的标量指标：turn 通道用 |角速度|，其余用 |速度|。
    此处用游戏直报值即可 —— 时序不同步不影响"是否还在变化"的判断。"""
    s = server.latest_states().get(ship_id)
    if s is None:
        return None
    return abs(s.ang_vel) if channel == "turn" else s.speed


def wait_plateau(server: TrainingServer, ship_id: str, channel: str) -> float:
    """施加动作期间等待进入平台期（顶到速度上限或阻尼稳态）。
    返回观测到的峰值指标。"""
    start = sim_now(server, ship_id)
    if start is None:
        return 0.0
    window: list[tuple[float, float]] = []   # (sim_t, metric)
    peak = 0.0
    deadline = time.time() + 180.0           # 真实时间兜底
    while time.time() < deadline:
        now = sim_now(server, ship_id)
        m = channel_metric(server, ship_id, channel)
        if now is None or m is None:
            time.sleep(0.05)
            continue
        peak = max(peak, m)
        window.append((now, m))
        window = [(t, v) for t, v in window if now - t <= PLATEAU_WINDOW]
        elapsed = now - start
        if elapsed >= ACT_TIME_MAX:
            break
        if elapsed >= ACT_TIME_MIN and len(window) >= 3:
            vals = [v for _, v in window]
            if max(vals) - min(vals) <= max(PLATEAU_REL_EPS * peak, 0.1):
                break                          # 已进入平台期
        time.sleep(0.05)
    return peak


def wait_decay(server: TrainingServer, ship_id: str, channel: str, peak: float) -> None:
    """松开动作后等待速度衰减到峰值的 COAST_STOP_RATIO 以下。"""
    start = sim_now(server, ship_id)
    if start is None:
        return
    stop_at = max(COAST_STOP_RATIO * peak, COAST_STOP_FLOOR)
    deadline = time.time() + 120.0
    while time.time() < deadline:
        now = sim_now(server, ship_id)
        m = channel_metric(server, ship_id, channel)
        if now is None or m is None:
            time.sleep(0.05)
            continue
        elapsed = now - start
        if elapsed >= COAST_TIME_MAX:
            return
        if elapsed >= COAST_TIME_MIN and m <= stop_at:
            return
        time.sleep(0.05)


def wait_full_stop(server: TrainingServer, ship_id: str, channel: str) -> None:
    """--drag 模式专用：松开后一直观测到完全停住（或超时）。"""
    start = sim_now(server, ship_id)
    if start is None:
        return
    deadline = time.time() + 2 * COAST_FULL_TIMEOUT
    while time.time() < deadline:
        now = sim_now(server, ship_id)
        m = channel_metric(server, ship_id, channel)
        if now is None or m is None:
            time.sleep(0.05)
            continue
        if now - start >= COAST_FULL_TIMEOUT or m <= COAST_FULL_STOP_EPS:
            return
        time.sleep(0.05)


# ---------------- 主流程 ----------------

def main() -> None:
    drag_mode = "--drag" in sys.argv
    actions = DRAG_ACTIONS if drag_mode else ACTIONS
    repeats = DRAG_REPEATS if drag_mode else REPEATS
    if drag_mode:
        print("[采样] --drag 阻力律测量模式：加速到上限后松开，记录到完全停住")

    server = TrainingServer()
    server.start()
    if not server.wait_connected(timeout=None):
        return

    ship_id = TARGET_SHIP_ID or server.wait_for_ship(timeout=120)
    if ship_id is None:
        print("[采样] 未等到任何舰船状态，退出")
        return
    print(f"[采样] 目标舰船: {ship_id}")

    recorder = Recorder(lambda: ship_id)
    server.set_frame_hook(recorder.hook)

    trial = 0
    total = len(actions) * repeats
    try:
        for channel, u in actions:
            for rep in range(repeats):
                trial += 1
                print(f"[采样] ({trial}/{total}) {channel} = {u:+.1f}  第 {rep + 1}/{repeats} 次")

                # 1) 归零并复位
                server.send_action(ship_id, 0.0, 0.0, 0.0)
                server.send_reset(ship_id)
                wait_settled(server, ship_id)

                # 2) 静止基线
                recorder.begin_trial(trial, channel)
                sim_sleep(server, ship_id, BASELINE_TIME)

                # 3) 施加动作，直到进入平台期（顶到速度上限或阻尼稳态）
                kwargs = {channel: u}
                server.send_action(ship_id, **kwargs)
                recorder.set_applied(u)
                peak = wait_plateau(server, ship_id, channel)

                # 4) 松开滑行
                server.send_action(ship_id, 0.0, 0.0, 0.0)
                recorder.set_applied(0.0)
                if drag_mode:
                    wait_full_stop(server, ship_id, channel)
                else:
                    wait_decay(server, ship_id, channel, peak)
                recorder.end_trial()
    except KeyboardInterrupt:
        print("\n[采样] 用户中断，保存已有数据")
    finally:
        server.send_action(ship_id, 0.0, 0.0, 0.0)
        server.set_frame_hook(None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "samples_drag_" if drag_mode else "samples_"
    path = os.path.join(OUTPUT_DIR, f"{prefix}{stamp}.csv")
    recorder.save(path)
    print(f"[采样] 完成。下一步运行: python analyze_fit.py {path}")


if __name__ == "__main__":
    main()
