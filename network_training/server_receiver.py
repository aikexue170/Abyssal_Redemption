import socket
import random

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

for line in reader:
    line = line.strip()
    if not line:
        continue
    if line == 'END':
        print(f"\n--- Frame ({len(frame)} ships) ---")
        for entry in frame:
            parts = entry.split(',')
            sid, x, y, vx, vy, a, av = parts[0], *[float(v) for v in parts[1:]]
            print(f"  {sid}: pos=({x:.0f},{y:.0f}) vel=({vx:.1f},{vy:.1f}) angle={a:.1f}")

        for entry in frame:
            sid = entry.split(',')[0]
            move = random.uniform(-1, 1)
            turn = random.uniform(-1, 1)
            strafe = random.uniform(-1, 1)
            writer.write(f"{sid},{move:.3f},{turn:.3f},{strafe:.3f}\n")
        writer.write("END\n")
        writer.flush()

        frame = []
    else:
        frame.append(line)

conn.close()
server.close()
