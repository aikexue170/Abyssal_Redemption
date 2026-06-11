import socket
import random
import time

HOST = '127.0.0.1'
PORT = 65432

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Server listening on {HOST}:{PORT}")

conn, addr = server.accept()
print(f"Connected: {addr}")

reader = conn.makefile('r')
writer = conn.makefile('w')
frame = []

last_time = time.time()

for line in reader:
    line = line.strip()
    if not line:
        continue
    if line == 'END':
        now = time.time()
        dt = now - last_time
        freq = 1.0 / dt if dt > 0 else 0
        last_time = now

        print(f"\n--- Frame ({len(frame)} ships) | 间隔 {dt:.3f}s | 频率 {freq:.1f} Hz ---")
        for entry in frame:
            parts = entry.split(',')
            idx, x, y, vx, vy, a, av = int(parts[0]), *[float(v) for v in parts[1:]]
            print(f"  #{idx}: pos=({x:.0f},{y:.0f}) vel=({vx:.1f},{vy:.1f}) angle={a:.1f}")

        for entry in frame:
            idx = entry.split(',')[0]
            move = random.uniform(-1, 1)
            turn = random.uniform(-1, 1)
            strafe = random.uniform(-1, 1)
            writer.write(f"{idx},{move:.3f},{turn:.3f},{strafe:.3f}\n")
        writer.write("END\n")
        writer.flush()

        frame = []
    else:
        frame.append(line)

conn.close()
server.close()
