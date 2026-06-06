package data.scripts.utils.PPO;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.concurrent.atomic.AtomicBoolean;

public class PPOClient_30state {
    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 65432;
    private static final float RAY_MAX_DISTANCE = 1f;

    // Socket缓冲区大小配置
    private static final int SOCKET_BUFFER_SIZE = 64 * 1024; // 64KB缓冲区
    private static final int SOCKET_TIMEOUT = 500; // 500ms超时

    // 目标位置和角度存储
    private float targetX = 0f;
    private float targetY = 0f;
    private float targetAngle = 0f;

    // 训练统计
    private int totalActions = 0;
    private int tacticalActions = 0;
    private long lastLogTime = 0;

    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;
    private final AtomicBoolean isConnected = new AtomicBoolean(false);

    // 单例模式
    private static class Holder {
        static final PPOClient_30state INSTANCE = new PPOClient_30state();
    }

    public static PPOClient_30state getInstance() {
        return Holder.INSTANCE;
    }

    public boolean isConnected() {
        return isConnected.get();
    }

    private PPOClient_30state() {} // 私有构造

    public synchronized void connect() {
        if (isConnected.get()) return;

        try {
            // 创建socket并设置缓冲区
            socket = new Socket(SERVER_IP, SERVER_PORT);

            // 设置发送和接收缓冲区大小
            socket.setSendBufferSize(SOCKET_BUFFER_SIZE);
            socket.setReceiveBufferSize(SOCKET_BUFFER_SIZE);
            socket.setTcpNoDelay(true); // 禁用Nagle算法，减少延迟
            socket.setSoTimeout(SOCKET_TIMEOUT); // 设置默认超时

            // 创建输出流和输入流
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            isConnected.set(true);

            // 重置统计
            totalActions = 0;
            tacticalActions = 0;
            lastLogTime = System.currentTimeMillis();

            // 输出连接信息（只输出一次）
            System.out.println("\n=== DQN LSTM 训练客户端已连接 ===");
            System.out.println("Socket缓冲区 - 发送: " + socket.getSendBufferSize() +
                    " bytes, 接收: " + socket.getReceiveBufferSize() + " bytes");
            System.out.println("实时状态监控开始...\n");

        } catch (IOException e) {
            System.err.println("连接失败: " + e.getMessage());
            cleanupResources();
        }
    }

    // 添加目标位置获取方法
    public Vector2f getTargetLocation() {
        return new Vector2f(targetX, targetY);
    }

    // 添加目标角度获取方法
    public float getTargetAngle() {
        return targetAngle;
    }

    public synchronized int getAction(Vector2f position,
                                      float angle,
                                      Vector2f velocity,
                                      float angularVelocity,
                                      float[] lasers,           // 16根射线距离
                                      boolean[] systemState) {  // 战术系统状态 [冷却中, 可用, 激活中]
        if (!isConnected.get()) return 0;

        CombatEngineAPI engine = Global.getCombatEngine();
        int actionToReturn = 0;

        try {
            // 1. 数据验证 - 检查NaN和无穷大（静默处理，不输出日志）
            if (Float.isNaN(position.x) || Float.isNaN(position.y) || Float.isNaN(angle) ||
                    Float.isNaN(velocity.x) || Float.isNaN(velocity.y) || Float.isNaN(angularVelocity)) {
                if (engine != null) {
                    engine.addFloatingText(position, "NaN输入，使用默认动作", 16f, Color.RED, null, 0.1f, 0.1f);
                }
                return 0;
            }

            // 检查无穷大
            if (Float.isInfinite(position.x) || Float.isInfinite(position.y) || Float.isInfinite(angle) ||
                    Float.isInfinite(velocity.x) || Float.isInfinite(velocity.y) || Float.isInfinite(angularVelocity)) {
                if (engine != null) {
                    engine.addFloatingText(position, "无穷大输入，使用默认动作", 16f, Color.RED, null, 0.1f, 0.1f);
                }
                return 0;
            }

            // 2. 构建状态数据字符串 (25个值，逗号分隔)
            StringBuilder stateData = new StringBuilder();

            // 基础状态 (6个) - 添加边界检查
            stateData.append(Math.max(-100000, Math.min(100000, position.x))).append(",");
            stateData.append(Math.max(-100000, Math.min(100000, position.y))).append(",");
            stateData.append(angle % 360).append(",");  // 规范化角度到0-360范围
            stateData.append(Math.max(-500, Math.min(500, velocity.x))).append(",");
            stateData.append(Math.max(-500, Math.min(500, velocity.y))).append(",");
            stateData.append(Math.max(-100, Math.min(100, angularVelocity))).append(",");

            // 射线数据 (16个) - 添加边界检查
            for (int i = 0; i < lasers.length; i++) {
                // 确保射线距离在合理范围内
                float safeLaser = Math.max(0, Math.min(RAY_MAX_DISTANCE, lasers[i]));
                if (Float.isNaN(safeLaser) || Float.isInfinite(safeLaser)) {
                    safeLaser = RAY_MAX_DISTANCE; // 使用最大距离作为安全值
                }
                stateData.append(safeLaser);
                if (i < lasers.length - 1) {
                    stateData.append(",");
                }
            }
            stateData.append(",");

            // 战术系统状态 (3个) - 将boolean转换为int (0或1)
            stateData.append(systemState[0] ? "1" : "0").append(",");
            stateData.append(systemState[1] ? "1" : "0").append(",");
            stateData.append(systemState[2] ? "1" : "0");

            // 3. 发送状态数据并立即刷新
            out.println(stateData.toString());
            out.flush();  // 重要：确保数据立即发送

            // 4. 接收响应（使用设置的超时）
            String response = in.readLine();

            if (response != null) {
                String[] parts = response.split(";");
                if (parts.length >= 4) {
                    // 5. 解析动作并验证范围
                    actionToReturn = Integer.parseInt(parts[0]);
                    if ((actionToReturn < 0 || actionToReturn > 7) && actionToReturn != 100) {
                        actionToReturn = 0;
                    }

                    // 6. 解析目标位置和角度
                    targetX = Float.parseFloat(parts[1]);
                    targetY = Float.parseFloat(parts[2]);
                    targetAngle = Float.parseFloat(parts[3]);

                    // 7. 验证目标数据
                    if (Float.isNaN(targetX) || Float.isNaN(targetY) || Float.isNaN(targetAngle)) {
                        targetX = position.x + 1000;
                        targetY = position.y;
                        targetAngle = 0;
                    }

                } else {
                    actionToReturn = 0;
                }
            } else {
                actionToReturn = 0;
            }

        } catch (java.net.SocketTimeoutException e) {
            actionToReturn = 0;
        } catch (NumberFormatException e) {
            actionToReturn = 0;
        } catch (Exception e) {
            actionToReturn = 0;
            disconnect();
        }

        // 8. 实时状态显示（不滚动刷新）
        displayRealTimeStatus(position, angle, velocity, angularVelocity,
                systemState, actionToReturn, engine);

        return actionToReturn;
    }

    private void displayRealTimeStatus(Vector2f position, float angle, Vector2f velocity,
                                       float angularVelocity, boolean[] systemState,
                                       int action, CombatEngineAPI engine) {
        // 更新统计
        totalActions++;
        if (action == 7) { // 假设7是战术动作
            tacticalActions++;
        }

        // 计算距离和速度
        float dx = targetX - position.x;
        float dy = targetY - position.y;
        float distance = (float) Math.sqrt(dx * dx + dy * dy);
        float speed = (float) Math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y);

        // 计算目标相对于飞船的方向
        float targetDirection = (float) Math.toDegrees(Math.atan2(dy, dx));
        float relativeDirection = targetDirection - angle;

        // 归一化到[-180, 180]
        while (relativeDirection > 180) relativeDirection -= 360;
        while (relativeDirection < -180) relativeDirection += 360;

        // 构建实时状态字符串
        StringBuilder status = new StringBuilder();
        status.append("🚀 飞船状态 | ");
        status.append(String.format("位置: (%.0f, %.0f) | ", position.x, position.y));
        status.append(String.format("距离目标: %.0f | ", distance));
        status.append(String.format("速度: %.1f | ", speed));
        status.append(String.format("角度: %.1f° | ", angle));
        status.append(String.format("相对目标: %.1f° | ", relativeDirection));
        status.append(String.format("动作: %d | ", action));
        status.append(String.format("战术: [CD:%s Avail:%s Active:%s] | ",
                systemState[0] ? "Y" : "N", systemState[1] ? "Y" : "N", systemState[2] ? "Y" : "N"));

        // 添加统计信息（每秒更新一次）
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastLogTime > 1000) {
            float tacticalRate = totalActions > 0 ? (float) tacticalActions / totalActions * 100 : 0;
            status.append(String.format("战术使用率: %.1f%%", tacticalRate));

            // 重置统计（每秒重置）
            tacticalActions = 0;
            totalActions = 0;
            lastLogTime = currentTime;
        }

        // 使用回车符实现不滚动刷新
        System.out.print("\r" + status.toString());
    }

    public synchronized void disconnect() {
        if (!isConnected.get()) return;
        isConnected.set(false);
        cleanupResources();
        System.out.println("\n\n=== 训练客户端已断开连接 ===");
    }

    private void cleanupResources() {
        try {
            if (out != null) out.close();
            if (in != null) in.close();
            if (socket != null) socket.close();
        } catch (IOException e) {
            // 静默处理资源清理错误
        }
    }
}
