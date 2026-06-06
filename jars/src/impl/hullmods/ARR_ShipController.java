package impl.hullmods;

import com.fs.starfarer.api.combat.BaseHullMod;
import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.ARR_EntityTimerManager;
import data.scripts.utils.ControlSystem.ShipControlSystem;
import data.scripts.utils.network.SimpleSocketClient;

public class ARR_ShipController extends BaseHullMod {
    ShipControlSystem shipControlSystem = null;
    SimpleSocketClient socketClient = null;

    // 当前控制指令
    private float currentMove = 0f;
    private float currentTurn = 0f;
    private float currentStrafe = 0f;

    // 服务器配置
    private static final String SERVER_IP = "localhost";
    private static final int SERVER_PORT = 8888;

    // 连接状态
    private boolean isConnected = false;
    private float lastConnectionAttempt = 0f;
    private static final float RECONNECT_INTERVAL = 5f; // 重连间隔（秒）

    private static final ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null || !ship.isAlive()) return;

        // 初始化控制系统
        if (shipControlSystem == null) {
            shipControlSystem = new ShipControlSystem(ship, amount);
        }

        shipControlSystem.updateTimer();

        // 初始化Socket客户端
        if (socketClient == null) {
            socketClient = new SimpleSocketClient(SERVER_IP, SERVER_PORT);
        }

        // 处理网络连接
        handleNetworkConnection(amount);

        // 处理接收到的命令
        handleReceivedCommands();

        // 应用当前控制指令
        applyControlCommands(amount);
    }

    private void handleNetworkConnection(float amount) {
        if (!isConnected) {
            lastConnectionAttempt += amount;
            // 定期尝试连接
            if (lastConnectionAttempt >= RECONNECT_INTERVAL) {
                lastConnectionAttempt = 0f;
                if (socketClient.connect()) {
                    isConnected = true;
                }
            }
        }
    }

    private void handleReceivedCommands() {
        if (!isConnected) return;

        // 处理所有接收到的消息
        while (socketClient.hasMessage()) {
            String message = socketClient.getMessage();
            if (message != null && !message.trim().isEmpty()) {
                parseAndExecuteCommand(message.trim());
            }
        }
    }

    private void parseAndExecuteCommand(String commandStr) {
        try {
            // 忽略连接欢迎消息
            if (commandStr.contains("Connected")) {
                return;
            }

            // 解析命令格式: move:0.5,turn:-0.3,strafe:0.2
            String[] parts = commandStr.split(",");

            for (String part : parts) {
                String[] keyValue = part.split(":");
                if (keyValue.length == 2) {
                    String key = keyValue[0].trim().toLowerCase();
                    float value = Float.parseFloat(keyValue[1].trim());

                    // 限制值在-1到1之间
                    value = Math.max(-1f, Math.min(1f, value));

                    switch (key) {
                        case "move":
                            currentMove = value;
                            break;
                        case "turn":
                            currentTurn = value;
                            break;
                        case "strafe":
                            currentStrafe = value;
                            break;
                    }
                }
            }


        } catch (Exception e) {
            System.err.println("解析远程命令失败: " + commandStr + " - " + e.getMessage());
        }
    }

    private void applyControlCommands(float amount) {
        // 应用移动控制
        if (Math.abs(currentMove) > 0.01f) {
            shipControlSystem.move(currentMove, amount);
        }

        // 应用转向控制
        if (Math.abs(currentTurn) > 0.01f) {
            shipControlSystem.turn(currentTurn, amount);
        }

        // 应用平移控制
        if (Math.abs(currentStrafe) > 0.01f) {
            shipControlSystem.strafe(currentStrafe, amount);
        }
    }
}