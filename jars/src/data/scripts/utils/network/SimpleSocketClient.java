package data.scripts.utils.network;

import java.io.*;
import java.net.Socket;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * 简单的Socket通讯客户端，用于调试和基本通讯
 */
public class SimpleSocketClient {
    private final String serverIP;
    private final int serverPort;

    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;

    private boolean isConnected = false;
    private Thread receiveThread;
    private final BlockingQueue<String> messageQueue = new LinkedBlockingQueue<>();

    // 连接配置
    private static final int SOCKET_TIMEOUT = 2000; // 2秒超时
    private static final int RECONNECT_DELAY = 1000; // 重连延迟1秒

    /**
     * 构造函数
     * @param serverIP 服务器IP地址
     * @param serverPort 服务器端口
     */
    public SimpleSocketClient(String serverIP, int serverPort) {
        this.serverIP = serverIP;
        this.serverPort = serverPort;
    }

    /**
     * 连接到服务器
     * @return 连接是否成功
     */
    public synchronized boolean connect() {
        if (isConnected) {
            System.out.println("Socket客户端: 已经连接到服务器");
            return true;
        }

        try {
            socket = new Socket(serverIP, serverPort);
            socket.setSoTimeout(SOCKET_TIMEOUT);
            socket.setTcpNoDelay(true); // 禁用Nagle算法

            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            isConnected = true;

            // 启动接收线程
            startReceiveThread();

            System.out.println("Socket客户端: 成功连接到 " + serverIP + ":" + serverPort);
            return true;

        } catch (IOException e) {
            System.err.println("Socket客户端: 连接失败 - " + e.getMessage());
            cleanup();
            return false;
        }
    }

    /**
     * 断开连接
     */
    public synchronized void disconnect() {
        if (!isConnected) return;

        System.out.println("Socket客户端: 断开连接");
        isConnected = false;
        cleanup();
    }

    /**
     * 发送消息
     * @param message 要发送的消息
     * @return 发送是否成功
     */
    public boolean send(String message) {
        if (!isConnected) {
            System.err.println("Socket客户端: 未连接，无法发送消息");
            return false;
        }

        try {
            out.println(message);
            return true;
        } catch (Exception e) {
            System.err.println("Socket客户端: 发送失败 - " + e.getMessage());
            disconnect();
            return false;
        }
    }

    /**
     * 发送格式化消息（自动添加时间戳）
     * @param format 格式化字符串
     * @param args 参数
     * @return 发送是否成功
     */
    public boolean sendFormatted(String format, Object... args) {
        String timestamp = String.format("[%.3f]", System.currentTimeMillis() / 1000.0);
        String message = timestamp + " " + String.format(format, args);
        return send(message);
    }

    /**
     * 检查是否有新消息
     * @return 如果有新消息返回true
     */
    public boolean hasMessage() {
        return !messageQueue.isEmpty();
    }

    /**
     * 获取最新消息（非阻塞）
     * @return 最新消息，如果没有返回null
     */
    public String getMessage() {
        return messageQueue.poll();
    }

    /**
     * 等待并获取消息（阻塞）
     * @param timeoutMs 超时时间（毫秒）
     * @return 消息，超时返回null
     */
    public String waitForMessage(long timeoutMs) {
        try {
            return messageQueue.poll(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    /**
     * 获取所有积压的消息
     * @return 消息数组
     */
    public String[] getAllMessages() {
        return messageQueue.toArray(new String[0]);
    }

    /**
     * 清空消息队列
     */
    public void clearMessages() {
        messageQueue.clear();
    }

    /**
     * 检查连接状态
     * @return 是否已连接
     */
    public boolean isConnected() {
        return isConnected;
    }

    /**
     * 获取服务器地址信息
     * @return 服务器地址字符串
     */
    public String getServerInfo() {
        return serverIP + ":" + serverPort;
    }

    // 私有方法

    private void startReceiveThread() {
        receiveThread = new Thread(() -> {
            System.out.println("Socket客户端: 启动接收线程");

            while (isConnected) {
                try {
                    String message = in.readLine();
                    if (message == null) {
                        // 服务器关闭连接
                        System.out.println("Socket客户端: 服务器关闭连接");
                        break;
                    }

                    // 将消息加入队列
                    messageQueue.offer(message);

                } catch (java.net.SocketTimeoutException e) {
                    // 超时是正常的，继续循环
                    continue;
                } catch (IOException e) {
                    System.err.println("Socket客户端: 接收错误 - " + e.getMessage());
                    break;
                }
            }

            // 循环结束，断开连接
            if (isConnected) {
                disconnect();
            }
        }, "SocketClient-ReceiveThread");

        receiveThread.setDaemon(true);
        receiveThread.start();
    }

    private void cleanup() {
        try {
            if (out != null) out.close();
            if (in != null) in.close();
            if (socket != null) socket.close();
        } catch (IOException e) {
            // 忽略清理时的错误
        }

        if (receiveThread != null && receiveThread.isAlive()) {
            receiveThread.interrupt();
        }

        out = null;
        in = null;
        socket = null;
        receiveThread = null;
    }
}