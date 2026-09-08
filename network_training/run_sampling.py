"""环境参数采样：向游戏内舰船施加一组标准动作并记录状态时间序列。

用法：
    1. 构建 jar，启动游戏，进入战斗，给被测舰船装上"训练桥接器"插件
    2. python run_sampling.py
    3. 采样完成后运行 analyze_fit.py 进行最小二乘拟合与可视化

流程：每个动作重复 REPEATS 次；每次先 RESET 归位 -> 静止基线 ->
      持续施加动作 ACT_TIME 秒 -> 松开滑行 COAST_TIME 秒。
      全部样本写入 output/samples_*.csv。
"""

from __future__ import annotations

import csv
import os
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
BASELINE_TIME = 0.5        # 施加动作前的静止基线（游戏内秒）
ACT_TIME = 3.0             # 动作持续时间（游戏内秒）
COAST_TIME = 2.0           # 松开后的滑行观测时间（游戏内秒）

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


# ---------------- 主流程 ----------------

def main() -> None:
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
    total = len(ACTIONS) * REPEATS
    try:
        for channel, u in ACTIONS:
            for rep in range(REPEATS):
                trial += 1
                print(f"[采样] ({trial}/{total}) {channel} = {u:+.1f}  第 {rep + 1}/{REPEATS} 次")

                # 1) 归零并复位
                server.send_action(ship_id, 0.0, 0.0, 0.0)
                server.send_reset(ship_id)
                wait_settled(server, ship_id)

                # 2) 静止基线
                recorder.begin_trial(trial, channel)
                sim_sleep(server, ship_id, BASELINE_TIME)

                # 3) 施加动作
                kwargs = {channel: u}
                server.send_action(ship_id, **kwargs)
                recorder.set_applied(u)
                sim_sleep(server, ship_id, ACT_TIME)

                # 4) 松开滑行
                server.send_action(ship_id, 0.0, 0.0, 0.0)
                recorder.set_applied(0.0)
                sim_sleep(server, ship_id, COAST_TIME)
                recorder.end_trial()
    except KeyboardInterrupt:
        print("\n[采样] 用户中断，保存已有数据")
    finally:
        server.send_action(ship_id, 0.0, 0.0, 0.0)
        server.set_frame_hook(None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"samples_{stamp}.csv")
    recorder.save(path)
    print(f"[采样] 完成。下一步运行: python analyze_fit.py {path}")


if __name__ == "__main__":
    main()
