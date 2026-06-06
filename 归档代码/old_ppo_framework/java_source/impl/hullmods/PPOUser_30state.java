package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.*;
import data.scripts.utils.ARR_EntityTimerManager;
import data.scripts.utils.ARR_Timer;
import data.scripts.utils.PPO.PPOClient_30state;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;
import java.util.List;

public class PPOUser_30state extends BaseHullMod {

    // 通信间隔（秒）
    private static final float COMMUNICATION_INTERVAL = 0.1f;

    // 计时器管理器实例
    private static final ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();

    // DQN客户端实例
    private static final PPOClient_30state DQN_CLIENT_LSTM = PPOClient_30state.getInstance();

    // 记录上次动作
    private int lastAction = 0;
    private boolean restartTag = false;
    private float[] lasers = new float[16];

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null || !ship.isAlive()) return;

        CombatEngineAPI engine = Global.getCombatEngine();
        if (engine == null) return;

        // ship.giveCommand(ShipCommand.FIRE, null, 1);

        // 获取全部武器
        List<WeaponAPI> weapons = ship.getAllWeapons();

        for(WeaponAPI weapon : weapons){
            weapon.setForceFireOneFrame(true);
        }
        // 获取或创建此舰船的计时器
        ARR_Timer timer = timerManager.getTimerForEntity(ship);

        // 检查是否达到通信间隔
        if (timerManager.isTargetReachedForEntity(ship, engine, COMMUNICATION_INTERVAL)) {
            // 重置计时器
            timerManager.resetTimerForEntity(ship);

            // 获取舰船状态
            Vector2f position = ship.getLocation();
            float angle = ship.getFacing();
            Vector2f velocity = ship.getVelocity();
            float angularVelocity = ship.getAngularVelocity();

            // 获取战术系统状态
            boolean systemCOOLDOWN = ship.getSystem().getState() == ShipSystemAPI.SystemState.COOLDOWN;
            boolean systemACTIVE = (ship.getSystem().getState() == ShipSystemAPI.SystemState.ACTIVE) ||
                    (ship.getSystem().getState() == ShipSystemAPI.SystemState.IN) ||
                    (ship.getSystem().getState() == ShipSystemAPI.SystemState.OUT);
            boolean systemIDLE = ship.getSystem().getState() == ShipSystemAPI.SystemState.IDLE;

            boolean[] tacticalState = new boolean[] {
                    systemCOOLDOWN,  // 冷却中
                    systemIDLE,      // 可用（不在冷却且不活跃）
                    systemACTIVE     // 激活中
            };

            // 获取激光数据
            for(int weaponTag = 0; weaponTag < 16; weaponTag++) {
                Object laserData = engine.getCustomData().get("laser" + weaponTag);
                if (laserData != null) {
                    lasers[weaponTag] = ((Number) laserData).floatValue();
                } else {
                    lasers[weaponTag] = 1.0f; // 默认值
                }
            }

            // 获取DQN决策
            int action = getDqnAction(position, angle, velocity, angularVelocity, lasers, tacticalState);

            // 检查重置指令（动作100表示重置）
            if (action == 100) {
                resetShip(ship);
                this.restartTag = false;
                // 跳过本次执行，等待下一帧
                return;
            }

            this.lastAction = action;

            // 显示调试信息
            // displayDebugInfo(ship, engine, position, angle, velocity, angularVelocity, action);

            // 执行动作
            executeAction(action, ship, engine);

            // 显示目标位置
            displayTargetInfo(engine);

        } else {
            // 持续应用上次动作
            executeAction(lastAction, ship, engine);
        }
    }

    private int getDqnAction(Vector2f position, float angle, Vector2f velocity,
                             float angularVelocity, float[] lasers, boolean[] tacticalState) {
        // 确保客户端连接
        if (!DQN_CLIENT_LSTM.isConnected()) {
            try {
                DQN_CLIENT_LSTM.connect();
            } catch (Exception e) {
                Global.getCombatEngine().addFloatingText(
                        position, "DQN连接失败: " + e.getMessage(), 20f, Color.RED, null, 1f, 1f
                );
                return 0;
            }
        }

        // 获取DQN决策
        return DQN_CLIENT_LSTM.getAction(position, angle, velocity, angularVelocity, lasers, tacticalState);
    }

    private void executeAction(int action, ShipAPI ship, CombatEngineAPI engine) {
        // 根据动作值执行操作
        switch (action) {
            case 0: // 无操作
                ship.blockCommandForOneFrame(ShipCommand.ACCELERATE);
                ship.blockCommandForOneFrame(ShipCommand.ACCELERATE_BACKWARDS);
                ship.blockCommandForOneFrame(ShipCommand.TURN_LEFT);
                ship.blockCommandForOneFrame(ShipCommand.TURN_RIGHT);
                ship.blockCommandForOneFrame(ShipCommand.STRAFE_LEFT);
                ship.blockCommandForOneFrame(ShipCommand.STRAFE_RIGHT);
                ship.blockCommandForOneFrame(ShipCommand.USE_SYSTEM);
                break;

            case 1: // W - 前进
                ship.blockCommandForOneFrame(ShipCommand.ACCELERATE_BACKWARDS);
                ship.giveCommand(ShipCommand.ACCELERATE, null, 0);
                break;

            case 2: // S - 后退
                ship.blockCommandForOneFrame(ShipCommand.ACCELERATE);
                ship.giveCommand(ShipCommand.ACCELERATE_BACKWARDS, null, 0);
                break;

            case 3: // A - 左转
                ship.blockCommandForOneFrame(ShipCommand.TURN_RIGHT);
                ship.giveCommand(ShipCommand.TURN_LEFT, null, 0);
                break;

            case 4: // D - 右转
                ship.blockCommandForOneFrame(ShipCommand.TURN_LEFT);
                ship.giveCommand(ShipCommand.TURN_RIGHT, null, 0);
                break;

            case 5: // Q - 左平移
                ship.blockCommandForOneFrame(ShipCommand.STRAFE_RIGHT);
                ship.giveCommand(ShipCommand.STRAFE_LEFT, null, 0);
                break;

            case 6: // E - 右平移
                ship.blockCommandForOneFrame(ShipCommand.STRAFE_LEFT);
                ship.giveCommand(ShipCommand.STRAFE_RIGHT, null, 0);
                break;

            case 7: // 减速
                ship.giveCommand(ShipCommand.DECELERATE, null, 0);
                break;

            default: // 未知动作，无操作
                ship.blockCommandForOneFrame(ShipCommand.ACCELERATE);
                ship.blockCommandForOneFrame(ShipCommand.TURN_LEFT);
                ship.blockCommandForOneFrame(ShipCommand.TURN_RIGHT);
                break;
        }
    }

    private void resetShip(ShipAPI ship) {
        ship.getLocation().set(0, 1500);
        ship.getVelocity().set(0, 0);
        ship.setAngularVelocity(0);
        ship.setFacing(90);
        this.restartTag = false;
    }

    public String getDescriptionParam(int index, ShipAPI.HullSize hullSize) {
        if (index == 0) return Float.toString(COMMUNICATION_INTERVAL);
        return null;
    }

    private void displayTargetInfo(CombatEngineAPI engine) {
        // 获取目标位置
        Vector2f targetLocation = DQN_CLIENT_LSTM.getTargetLocation();
        float targetAngle = DQN_CLIENT_LSTM.getTargetAngle();

        // 在状态栏显示目标信息
        engine.maintainStatusForPlayerShip("3", null, "目标位置",
                String.format("(%.0f, %.0f)", targetLocation.x, targetLocation.y), true);
        engine.maintainStatusForPlayerShip("4", null, "目标角度",
                String.format("%.1f°", targetAngle), true);

        // 创建目标标记粒子效果
        engine.addSmoothParticle(
                targetLocation, new Vector2f(0, 0), 100f, 1f, 0.1f, Color.red
        );
        engine.addSmoothParticle(
                targetLocation, new Vector2f(0, 0), 50f, 1f, 0.1f, Color.white
        );

        // 显示目标方向线
        Vector2f targetDirection = new Vector2f(
                (float)Math.cos(Math.toRadians(targetAngle)),
                (float)Math.sin(Math.toRadians(targetAngle))
        );
        targetDirection.scale(200f); // 缩放箭头长度

        Vector2f arrowEnd = new Vector2f(targetLocation.x + targetDirection.x,
                targetLocation.y + targetDirection.y);

        engine.addSmoothParticle(arrowEnd, new Vector2f(0, 0), 30f, 1f, 0.1f, Color.green);
    }
}