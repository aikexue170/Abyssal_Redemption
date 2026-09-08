"""Java <-> Python 行文本协议编解码。

协议（TCP，行文本，一帧多条记录，以 END 结尾）：
    Java -> Python:
        STATE,<simTime>,<shipId>,<x>,<y>,<vx>,<vy>,<facing>,<angVel>
        END
    Python -> Java:
        ACT,<shipId>,<move>,<turn>,<strafe>   # 三轴开度 [-1,1]
        RESET,<shipId|ALL>                     # 复位到出生点并清零速度/角速度
        END
"""

from __future__ import annotations

from dataclasses import dataclass, field

END = "END"


@dataclass
class ShipState:
    sim_time: float
    ship_id: str
    x: float
    y: float
    vx: float          # 游戏直接上报的速度（与位置存在时序不同步，拟合勿用）
    vy: float
    facing: float      # 度
    ang_vel: float     # 度/秒

    @property
    def speed(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5


def parse_state_line(line: str) -> ShipState | None:
    """解析一行 STATE 记录；非 STATE 或格式错误返回 None。"""
    parts = line.strip().split(",")
    if len(parts) != 9 or parts[0] != "STATE":
        return None
    try:
        return ShipState(
            sim_time=float(parts[1]),
            ship_id=parts[2],
            x=float(parts[3]),
            y=float(parts[4]),
            vx=float(parts[5]),
            vy=float(parts[6]),
            facing=float(parts[7]),
            ang_vel=float(parts[8]),
        )
    except ValueError:
        return None


def encode_act(ship_id: str, move: float, turn: float, strafe: float) -> str:
    clamp = lambda v: max(-1.0, min(1.0, float(v)))
    return f"ACT,{ship_id},{clamp(move):.4f},{clamp(turn):.4f},{clamp(strafe):.4f}"


def encode_reset(ship_id: str = "ALL") -> str:
    return f"RESET,{ship_id}"
