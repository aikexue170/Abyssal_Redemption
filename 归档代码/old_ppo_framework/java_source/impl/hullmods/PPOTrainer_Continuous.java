package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.*;
import data.scripts.utils.ARR_EntityTimerManager;
import data.scripts.utils.ARR_TemporalShellUtil;
import data.scripts.utils.ARR_Timer;
import data.scripts.utils.ControlSystem.ShipControlSystem;
import data.scripts.utils.PPO.SimpleSocketClient;
import org.lwjgl.util.vector.Vector2f;

public class PPOTrainer_Continuous extends BaseHullMod {

    // 通信间隔（秒）
    private static final float COMMUNICATION_INTERVAL = 0.01f;

    // 计时器管理器实例
    private static final ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();

    // Socket客户端实例
    private SimpleSocketClient socketClient = null;

    // 当前控制指令
    private float currentMove = 0f;
    private float currentTurn = 0f;
    private float currentStrafe = 0f;

    // 服务器配置
    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 8888;

    // 连接状态
    private boolean isConnected = false;
    private float lastConnectionAttempt = 0f;
    private static final float RECONNECT_INTERVAL = 5f; // 重连间隔（秒）

    // 控制系统实例
    private ShipControlSystem shipControlSystem = null;

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null || !ship.isAlive()) return;

        // 初始化控制系统
        if (shipControlSystem == null) {
            shipControlSystem = new ShipControlSystem(ship, amount);
        }

        shipControlSystem.updateTimer();

        //ARR_TemporalShellUtil.applyTemporalShell(ship, 10f, false);
        CombatEngineAPI engine = Global.getCombatEngine();
        if (engine == null) return;

        // 强制所有武器开火
        for(WeaponAPI weapon : ship.getAllWeapons()){
            weapon.setForceFireOneFrame(true);
        }

        // 初始化Socket客户端
        if (socketClient == null) {
            socketClient = new SimpleSocketClient(SERVER_IP, SERVER_PORT);
        }

        // 处理网络连接
        handleNetworkConnection(amount);

        // 获取或创建此舰船的计时器
        ARR_Timer timer = timerManager.getTimerForEntity(ship);

        // 检查是否达到通信间隔
        if (timerManager.isTargetReachedForEntity(ship, engine, COMMUNICATION_INTERVAL)) {
            // 重置计时器
            timerManager.resetTimerForEntity(ship);

            // 收集所有状态数据
            String stateData = collectAllStateData(ship, engine);

            // 发送状态数据到Python端
            if (isConnected) {
                socketClient.send(stateData);
            }

            // 处理接收到的动作命令
            handleReceivedCommands();

            // 显示调试信息
            displayDebugInfo(ship, engine);

            // 显示目标位置
            displayTargetInfo(engine);

        }

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
                    System.out.println("PPO训练器: 成功连接到Python服务器");
                }
            }
        }
    }

    private String collectAllStateData(ShipAPI ship, CombatEngineAPI engine) {
        // 收集所有可能的状态数据，由Python端决定如何使用

        // 基本位置和速度信息
        Vector2f position = ship.getLocation();
        float angle = ship.getFacing();
        Vector2f velocity = ship.getVelocity();
        float angularVelocity = ship.getAngularVelocity();

        // 战术系统状态
        boolean systemCOOLDOWN = ship.getSystem().getState() == ShipSystemAPI.SystemState.COOLDOWN;
        boolean systemACTIVE = (ship.getSystem().getState() == ShipSystemAPI.SystemState.ACTIVE) ||
                (ship.getSystem().getState() == ShipSystemAPI.SystemState.IN) ||
                (ship.getSystem().getState() == ShipSystemAPI.SystemState.OUT);
        boolean systemIDLE = ship.getSystem().getState() == ShipSystemAPI.SystemState.IDLE;

        // 激光数据
        float[] lasers = new float[16];
        for(int weaponTag = 0; weaponTag < 16; weaponTag++) {
            Object laserData = engine.getCustomData().get("laser" + weaponTag);
            if (laserData != null) {
                lasers[weaponTag] = ((Number) laserData).floatValue();
            } else {
                lasers[weaponTag] = 1.0f; // 默认值
            }
        }

        // 构建状态数据字符串
        // 格式：所有数据用逗号分隔，Python端可以自由选择使用哪些数据
        StringBuilder stateBuilder = new StringBuilder();

        // 位置和角度
        stateBuilder.append(position.x).append(",");
        stateBuilder.append(position.y).append(",");
        stateBuilder.append(angle).append(",");

        // 速度信息
        stateBuilder.append(velocity.x).append(",");
        stateBuilder.append(velocity.y).append(",");
        stateBuilder.append(angularVelocity).append(",");

        // 战术系统状态
        stateBuilder.append(systemCOOLDOWN ? "1" : "0").append(",");
        stateBuilder.append(systemIDLE ? "1" : "0").append(",");
        stateBuilder.append(systemACTIVE ? "1" : "0").append(",");

        // 激光数据
        for (int i = 0; i < lasers.length; i++) {
            stateBuilder.append(lasers[i]);
            if (i < lasers.length - 1) {
                stateBuilder.append(",");
            }
        }

        return stateBuilder.toString();
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

            System.out.println(String.format(
                    "接收到动作命令: move=%.2f, turn=%.2f, strafe=%.2f",
                    currentMove, currentTurn, currentStrafe
            ));

        } catch (Exception e) {
            System.err.println("解析PPO动作命令失败: " + commandStr + " - " + e.getMessage());
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

    private void displayDebugInfo(ShipAPI ship, CombatEngineAPI engine) {
        Vector2f position = ship.getLocation();

        // 使用System.out输出调试信息
        System.out.println(String.format(
                "Ship: (%.1f, %.1f), Vel: (%.1f, %.1f), Actions: M=%.2f, T=%.2f, S=%.2f",
                position.x, position.y,
                ship.getVelocity().x, ship.getVelocity().y,
                currentMove, currentTurn, currentStrafe
        ));
    }

    private void displayTargetInfo(CombatEngineAPI engine) {
        // 这里可以添加目标显示逻辑
        // 由于Python端决定目标位置，可能需要从Python端接收目标信息
        // 暂时留空，等Python端实现后再补充
    }

    public String getDescriptionParam(int index, ShipAPI.HullSize hullSize) {
        if (index == 0) return Float.toString(COMMUNICATION_INTERVAL);
        return null;
    }

    @Override
    public void applyEffectsAfterShipCreation(ShipAPI ship, String id) {
        // 初始化代码
        System.out.println("PPO连续动作训练器已加载");
    }

}