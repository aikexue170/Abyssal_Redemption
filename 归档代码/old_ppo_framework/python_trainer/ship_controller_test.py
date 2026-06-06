import socket
import threading
import time
import json


class ShipControlServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.client_conn = None
        self.client_addr = None
        self.running = False
        self.current_commands = {
            'move': 0.0,
            'turn': 0.0,
            'strafe': 0.0
        }

    def start_server(self):
        """启动服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)  # 设置超时以便可以检查停止标志

            print(f"飞船控制服务器启动在 {self.host}:{self.port}")
            print("等待游戏连接...")

            self.running = True
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()

            # 启动命令输入线程
            self.input_thread = threading.Thread(target=self._command_input)
            self.input_thread.daemon = True
            self.input_thread.start()

            return True

        except Exception as e:
            print(f"启动服务器失败: {e}")
            return False

    def _accept_connections(self):
        """接受客户端连接"""
        while self.running:
            try:
                conn, addr = self.socket.accept()
                print(f"游戏已连接: {addr}")

                # 如果已有连接，关闭旧的
                if self.client_conn:
                    try:
                        self.client_conn.close()
                    except:
                        pass

                self.client_conn = conn
                self.client_addr = addr
                self.client_conn.settimeout(1.0)

                # 发送欢迎消息
                welcome_msg = "Connected to Ship Control Server"
                self._send_message(welcome_msg)

                # 启动接收线程
                recv_thread = threading.Thread(target=self._receive_messages)
                recv_thread.daemon = True
                recv_thread.start()

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"接受连接时出错: {e}")

    def _receive_messages(self):
        """接收来自游戏的消息"""
        while self.running and self.client_conn:
            try:
                data = self.client_conn.recv(1024).decode('utf-8')
                if not data:
                    print("游戏断开连接")
                    break

                # 处理接收到的消息
                print(f"来自游戏: {data}")

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"接收消息时出错: {e}")
                    break

        # 清理连接
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
            self.client_conn = None
            self.client_addr = None

    def _command_input(self):
        """处理用户输入命令"""
        print("\n=== 飞船控制指令 ===")
        print("格式: move:0.5,turn:-0.3,strafe:0.2")
        print("move: 前进(+)后退(-), -1到1")
        print("turn: 左转(-)右转(+), -1到1")
        print("strafe: 左平移(-)右平移(+), -1到1")
        print("输入 'stop' 停止所有控制")
        print("输入 'exit' 退出服务器")
        print("===================\n")

        while self.running:
            try:
                command = input("请输入控制指令: ").strip()

                if command.lower() == 'exit':
                    self.stop_server()
                    break
                elif command.lower() == 'stop':
                    self.current_commands = {'move': 0.0, 'turn': 0.0, 'strafe': 0.0}
                    self._send_commands()
                    print("已停止所有控制")
                else:
                    if self._parse_command(command):
                        self._send_commands()
                    else:
                        print("指令格式错误，请使用: move:0.5,turn:-0.3,strafe:0.2")

            except EOFError:
                break
            except Exception as e:
                print(f"输入处理错误: {e}")

    def _parse_command(self, command_str):
        """解析命令字符串"""
        try:
            # 重置命令
            new_commands = {'move': 0.0, 'turn': 0.0, 'strafe': 0.0}

            parts = command_str.split(',')
            for part in parts:
                key_value = part.split(':')
                if len(key_value) == 2:
                    key = key_value[0].strip().lower()
                    value = float(key_value[1].strip())

                    # 限制值在-1到1之间
                    value = max(-1.0, min(1.0, value))

                    if key in new_commands:
                        new_commands[key] = value

            # 更新当前命令
            self.current_commands.update(new_commands)
            return True

        except Exception as e:
            print(f"解析命令失败: {e}")
            return False

    def _send_commands(self):
        """发送当前命令到游戏"""
        command_str = f"move:{self.current_commands['move']:.2f},turn:{self.current_commands['turn']:.2f},strafe:{self.current_commands['strafe']:.2f}"
        self._send_message(command_str)
        print(f"发送指令: {command_str}")

    def _send_message(self, message):
        """发送消息到客户端"""
        if self.client_conn:
            try:
                self.client_conn.sendall((message + '\n').encode('utf-8'))
                return True
            except Exception as e:
                print(f"发送消息失败: {e}")
                self.client_conn = None
        else:
            print("没有连接的客户端")
        return False

    def stop_server(self):
        """停止服务器"""
        print("正在停止服务器...")
        self.running = False

        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        print("服务器已停止")


def main():
    server = ShipControlServer()

    if not server.start_server():
        return

    try:
        # 保持主线程运行
        while server.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n收到中断信号")
    finally:
        server.stop_server()


if __name__ == "__main__":
    main()