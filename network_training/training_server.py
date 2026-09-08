"""TCP 服务器：等待 Java 端（ARR_TrainingBridge）连入，接收状态帧、下发指令。"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Optional

from protocol import END, ShipState, encode_act, encode_reset, parse_state_line


class TrainingServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 65432):
        self.host = host
        self.port = port

        self._conn: Optional[socket.socket] = None
        self._writer = None
        self._lock = threading.Lock()

        # ship_id -> 最新一帧状态
        self._latest: dict[str, ShipState] = {}
        # 每收到一个完整帧（END）回调一次，参数为本帧所有状态
        self._frame_hook: Optional[Callable[[list[ShipState]], None]] = None

        self.connected = threading.Event()

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """绑定端口并在后台线程等待 Java 端连入。"""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[TrainingServer] 监听 {self.host}:{self.port}，等待游戏连入...")
        threading.Thread(target=self._accept_loop, args=(srv,), daemon=True).start()

    def wait_connected(self, timeout: float | None = None) -> bool:
        return self.connected.wait(timeout)

    def close(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except OSError:
            pass

    # ---------- 连接与接收 ----------

    def _accept_loop(self, srv: socket.socket) -> None:
        conn, addr = srv.accept()
        print(f"[TrainingServer] 已连接: {addr}")
        with self._lock:
            self._conn = conn
            self._writer = conn.makefile("w")
        self.connected.set()

        frame: list[ShipState] = []
        try:
            for raw in conn.makefile("r"):
                line = raw.strip()
                if not line:
                    continue
                if line == END:
                    states = frame
                    frame = []
                    with self._lock:
                        for s in states:
                            self._latest[s.ship_id] = s
                        hook = self._frame_hook
                    if hook and states:
                        hook(states)
                else:
                    s = parse_state_line(line)
                    if s is not None:
                        frame.append(s)
        except OSError as e:
            print(f"[TrainingServer] 连接断开: {e}")
        finally:
            self.connected.clear()

    # ---------- 状态查询 ----------

    def set_frame_hook(self, hook: Optional[Callable[[list[ShipState]], None]]) -> None:
        with self._lock:
            self._frame_hook = hook

    def latest_states(self) -> dict[str, ShipState]:
        with self._lock:
            return dict(self._latest)

    def wait_for_ship(self, timeout: float = 60.0) -> Optional[str]:
        """等待第一艘上报状态的舰船，返回其 ship_id。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            states = self.latest_states()
            if states:
                return sorted(states)[0]
            time.sleep(0.2)
        return None

    # ---------- 指令下发 ----------

    def _send(self, *lines: str) -> None:
        with self._lock:
            writer = self._writer
        if writer is None:
            return
        try:
            for line in lines:
                writer.write(line + "\n")
            writer.write(END + "\n")
            writer.flush()
        except OSError as e:
            print(f"[TrainingServer] 发送失败: {e}")

    def send_action(self, ship_id: str, move: float = 0.0, turn: float = 0.0, strafe: float = 0.0) -> None:
        self._send(encode_act(ship_id, move, turn, strafe))

    def send_reset(self, ship_id: str = "ALL") -> None:
        self._send(encode_reset(ship_id))
