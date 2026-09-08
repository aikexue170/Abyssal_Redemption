package data.scripts.utils.ControlSystem;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import com.fs.starfarer.api.combat.ShipCommand;
import data.scripts.utils.ARR_Timer;
import org.lwjgl.util.vector.Vector2f;

import static org.lazywizard.lazylib.combat.CombatUtils.applyForce;

/**
 * 舰船运动控制系统：三轴（前进/后退、左右旋转、左右平移）开度控制。
 * 所有输入均为 [-1,1] 的无级开度。
 *
 * 注意：本类的 MAX_* / *_ACCELERATION 只是标称值，
 * 游戏内真实运动参数由 network_training 下的采样拟合流程测得。
 */
public class ShipControlSystem {
    private static final float DEADZONE = 0.01f;

    private final ShipAPI ship;
    private float amount;
    private final ARR_Timer timer;
    private final CombatEngineAPI engine = Global.getCombatEngine();

    private final float ACCELERATION;
    private final float TURN_ACCELERATION;
    private final float STRAFE_ACCELERATION;
    private final float MAX_FORWARD_SPEED;
    private final float MAX_BACKWARD_SPEED;
    private final float MAX_TURN_SPEED;
    private final float MAX_STRAFE_SPEED;

    public ShipControlSystem(ShipAPI ship, float amount){
        this.ship = ship;
        // 把控制器绑定到舰船 customData 中
        ship.getCustomData().put("shipControlSystem", this);
        this.amount = amount;
        this.timer = new ARR_Timer();
        MAX_STRAFE_SPEED = 15f;
        MAX_TURN_SPEED = 20f;
        MAX_BACKWARD_SPEED = 20f;
        MAX_FORWARD_SPEED = 40f;
        STRAFE_ACCELERATION = 300f;
        TURN_ACCELERATION = 20f;
        ACCELERATION = 500f;
    }

    // 获取舰船控制器的静态方法
    public static ShipControlSystem getControlSystem(ShipAPI ship){
        return (ShipControlSystem) ship.getCustomData().get("shipControlSystem");
    }

    public ARR_Timer getTimer() {
        return timer;
    }

    public void updateTimer(){
        timer.timer(engine);
    }

    /**
     * 前进/后退控制。
     * @param throttle 开度，正值前进，负值后退，范围 [-1,1]
     * @param amount   帧时间因子，保证不同帧率下加速度一致
     */
    public void move(float throttle, float amount) {
        throttle = clamp(throttle);
        if (Math.abs(throttle) < DEADZONE) return;

        Vector2f velocity = new Vector2f(ship.getVelocity());
        float facingRad = (float) Math.toRadians(ship.getFacing());

        // 船头方向单位向量（0度=右，90度=上）
        Vector2f forwardDir = new Vector2f(
                (float) Math.cos(facingRad),
                (float) Math.sin(facingRad)
        );

        // 当前速度在船头方向上的投影
        float speedInForwardDirection = Vector2f.dot(velocity, forwardDir);

        float moveDirection;
        float speedLimit;
        if (throttle > 0) {
            moveDirection = ship.getFacing();
            speedLimit = MAX_FORWARD_SPEED;
            ship.giveCommand(ShipCommand.ACCELERATE, null, 0);
        } else {
            moveDirection = ship.getFacing() + 180f;
            speedLimit = -MAX_BACKWARD_SPEED; // 负值：速度投影下限
            ship.giveCommand(ShipCommand.ACCELERATE_BACKWARDS, null, 0);
        }

        // 未达到速度阈值才继续施力
        boolean canMove = (throttle > 0 && speedInForwardDirection < speedLimit)
                || (throttle < 0 && speedInForwardDirection > speedLimit);

        if (canMove) {
            applyForce(ship, moveDirection, ACCELERATION * Math.abs(throttle) * amount);
        }
    }

    /**
     * 转向控制。
     * @param direction -1 到 1，负值左转，正值右转
     */
    public void turn(float direction, float amount) {
        direction = clamp(direction);
        if (Math.abs(direction) < DEADZONE) return;

        float newAngularVelocity = ship.getAngularVelocity() + TURN_ACCELERATION * direction * amount;
        if (Math.abs(newAngularVelocity) > MAX_TURN_SPEED) {
            newAngularVelocity = Math.signum(newAngularVelocity) * MAX_TURN_SPEED;
        }
        ship.setAngularVelocity(newAngularVelocity);
    }

    /**
     * 平移控制。
     * @param throttle 开度，正值向右平移，负值向左平移，范围 [-1,1]
     */
    public void strafe(float throttle, float amount) {
        throttle = clamp(throttle);
        if (Math.abs(throttle) < DEADZONE) return;

        Vector2f velocity = new Vector2f(ship.getVelocity());
        float facingRad = (float) Math.toRadians(ship.getFacing());

        // 右舷方向单位向量（船头方向 +90 度）
        Vector2f strafeDir = new Vector2f(
                (float) Math.cos(facingRad + Math.PI / 2),
                (float) Math.sin(facingRad + Math.PI / 2)
        );

        float speedInStrafeDirection = Vector2f.dot(velocity, strafeDir);

        float strafeDirection;
        float speedLimit;
        if (throttle > 0) {
            strafeDirection = ship.getFacing() + 90f;
            speedLimit = MAX_STRAFE_SPEED;
        } else {
            strafeDirection = ship.getFacing() - 90f;
            speedLimit = -MAX_STRAFE_SPEED;
        }

        boolean canStrafe = (throttle > 0 && speedInStrafeDirection < speedLimit)
                || (throttle < 0 && speedInStrafeDirection > speedLimit);

        if (canStrafe) {
            applyForce(ship, strafeDirection, STRAFE_ACCELERATION * Math.abs(throttle) * amount);
        }
    }

    private static float clamp(float v) {
        return Math.max(-1f, Math.min(1f, v));
    }
}
