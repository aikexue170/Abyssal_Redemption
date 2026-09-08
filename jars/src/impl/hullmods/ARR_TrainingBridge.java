package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.BaseHullMod;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.ControlSystem.ShipControlSystem;
import data.scripts.utils.network_training.ARR_TrainingClient;
import org.lwjgl.util.vector.Vector2f;

import java.util.HashMap;
import java.util.Map;

/**
 * 训练桥接器：本 Mod 强化学习链路的唯一入口（hullmod）。
 *
 * 每艘装配此插件的舰船都会被注册；
 * 每 0.1 秒（游戏内时间）将所有已注册舰船的状态发送给 Python 训练端，
 * 同时接收动作指令（ACT）并施加到对应舰船的 ShipControlSystem 上，
 * 或响应复位指令（RESET）将舰船归位并清零速度/角速度。
 */
public class ARR_TrainingBridge extends BaseHullMod {

    private static final float SEND_INTERVAL = 0.1f;      // 状态发送间隔（秒）
    private static final float RECONNECT_INTERVAL = 5f;   // 断线重连间隔（秒）

    // ---- 所有实例共享的静态通信状态 ----
    private static final Map<String, ARR_TrainingBridge> INSTANCES = new HashMap<>();
    private static ARR_TrainingClient client = null;
    private static CombatEngineAPI lastEngine = null;
    private static float lastSendTime = -999f;
    private static float lastReconnectAttempt = -999f;

    // ---- 每艘舰船自己的状态 ----
    private ShipAPI ship;
    private String shipId;
    private ShipControlSystem controlSystem;

    // 出生姿态（用于 RESET 复位）
    private boolean initialPoseCaptured = false;
    private final Vector2f initialLocation = new Vector2f();
    private float initialFacing;

    // 当前动作指令，范围 [-1,1]，由 Python 端写入
    private volatile float cmdMove = 0f;
    private volatile float cmdTurn = 0f;
    private volatile float cmdStrafe = 0f;

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null) return;

        CombatEngineAPI engine = Global.getCombatEngine();
        if (engine == null) return;

        // 进入新一场战斗时清空上一场的注册表
        if (engine != lastEngine) {
            INSTANCES.clear();
            lastEngine = engine;
        }

        if (!ship.isAlive()) {
            INSTANCES.remove(ship.getId());
            return;
        }

        this.ship = ship;
        this.shipId = ship.getId();
        INSTANCES.put(shipId, this);

        if (controlSystem == null) {
            controlSystem = new ShipControlSystem(ship, amount);
        }

        // 首次运行时记录出生姿态
        if (!initialPoseCaptured) {
            initialLocation.set(ship.getLocation());
            initialFacing = ship.getFacing();
            initialPoseCaptured = true;
        }

        ensureConnection(engine);

        float now = engine.getTotalElapsedTime(false);
        if (now - lastSendTime >= SEND_INTERVAL) {
            lastSendTime = now;
            sendAllStates(now);
            dispatchMessages();
        }

        applyCommands(amount);
    }

    private static void ensureConnection(CombatEngineAPI engine) {
        if (client == null) {
            client = new ARR_TrainingClient();
        }
        if (!client.isConnected()) {
            float now = engine.getTotalElapsedTime(false);
            if (now - lastReconnectAttempt >= RECONNECT_INTERVAL) {
                lastReconnectAttempt = now;
                client.connect();
            }
        }
    }

    private void sendAllStates(float now) {
        if (client == null || !client.isConnected()) return;
        for (ARR_TrainingBridge bridge : INSTANCES.values()) {
            if (bridge.ship != null && bridge.ship.isAlive()) {
                client.sendShipState(bridge.ship, now);
            }
        }
        client.endStateFrame();
    }

    private void dispatchMessages() {
        if (client == null || !client.isConnected()) return;

        ARR_TrainingClient.Message msg;
        while ((msg = client.nextMessage()) != null) {
            switch (msg.type) {
                case ACT: {
                    ARR_TrainingBridge target = INSTANCES.get(msg.shipId);
                    if (target != null) {
                        target.cmdMove = msg.move;
                        target.cmdTurn = msg.turn;
                        target.cmdStrafe = msg.strafe;
                    }
                    break;
                }
                case RESET: {
                    if (msg.shipId.equalsIgnoreCase("ALL")) {
                        for (ARR_TrainingBridge b : INSTANCES.values()) {
                            b.resetToInitialPose();
                        }
                    } else {
                        ARR_TrainingBridge target = INSTANCES.get(msg.shipId);
                        if (target != null) {
                            target.resetToInitialPose();
                        }
                    }
                    break;
                }
            }
        }
    }

    private void applyCommands(float amount) {
        if (controlSystem == null) return;
        controlSystem.move(cmdMove, amount);
        controlSystem.turn(cmdTurn, amount);
        controlSystem.strafe(cmdStrafe, amount);
    }

    /** 复位：回到出生位置与朝向，清零线速度/角速度，并清空当前动作。 */
    private void resetToInitialPose() {
        if (ship == null || !initialPoseCaptured) return;

        ship.getLocation().set(initialLocation.x, initialLocation.y);
        ship.setFacing(initialFacing);
        ship.getVelocity().set(0f, 0f);
        ship.setAngularVelocity(0f);

        cmdMove = 0f;
        cmdTurn = 0f;
        cmdStrafe = 0f;
    }
}
