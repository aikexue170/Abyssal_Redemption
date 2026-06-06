package data.scripts.utils.ControlSystem;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import com.fs.starfarer.api.combat.ShipCommand;
import data.scripts.utils.ARR_Timer;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;

import static org.lazywizard.lazylib.combat.CombatUtils.applyForce;

public class ShipControlSystem {
    ShipAPI ship;
    float amount;
    ARR_Timer timer;
    CombatEngineAPI engine = Global.getCombatEngine();

    private final float ACCELERATION;
    private final float TURN_ACCELERATION;
    private final float STRAFE_ACCELERATION;
    private final float MAX_FORWARD_SPEED;
    private final float MAX_BACKWARD_SPEED;
    private final float MAX_TURN_SPEED;
    private final float MAX_STRAFE_SPEED;

    public ShipControlSystem(ShipAPI ship, float amount){
        this.ship = ship;
        // 把控制器绑定到舰船customData中
        ship.getCustomData().put("shipControlSystem", this);
        this.amount = amount;
        timer = new ARR_Timer();
        MAX_STRAFE_SPEED = 15f;
        MAX_TURN_SPEED = 20f;
        MAX_BACKWARD_SPEED = 20f;
        MAX_FORWARD_SPEED = 40f;
        STRAFE_ACCELERATION = 300f;
        TURN_ACCELERATION = 20f;
        ACCELERATION = 500f;
    }

    // 获取舰船控制器的静态方法
    static public ShipControlSystem getControlSystem(ShipAPI ship){
        return (ShipControlSystem) ship.getCustomData().get("shipControlSystem");
    }

    public ARR_Timer getTimer() {
        return timer;
    }

    public void updateTimer(){
        timer.timer(engine);
    }

    // 可控制开度的移动方法
    // throttle: 开度，正值表示前进，负值表示后退，范围[-1,1]
    // amount: 时间因子，用于确保不同帧率下加速度一致
    public void move(float throttle, float amount) {
        System.out.println("throttle: " + throttle);
        ship.giveCommand(ShipCommand.ACCELERATE, null, 0);
        // 限制开度范围在[-1,1]
        throttle = Math.max(-1f, Math.min(1f, throttle));

        // 如果开度为0，不执行任何操作
        if (throttle == 0) return;

        Vector2f velocity = new Vector2f(ship.getVelocity());

        // 获取船头方向的角度（以弧度表示）
        float facingRad = (float)Math.toRadians(ship.getFacing());

        // 修复：正确的船头方向单位向量计算
        // Starfarer坐标系：0度=右，90度=上，180度=左，270度=下
        Vector2f forwardDir = new Vector2f(
                (float)Math.cos(facingRad),  // x分量 - 使用cos
                (float)Math.sin(facingRad)   // y分量 - 使用sin
        );

        // 计算速度在船头方向上的投影
        float speedInForwardDirection = Vector2f.dot(velocity, forwardDir);

        // 根据开度正负确定移动方向和速度限制
        float moveDirection;
        float speedLimit;
        String moveText;

        if (throttle > 0) {
            // 前进
            moveDirection = ship.getFacing();
            speedLimit = MAX_FORWARD_SPEED;
            ship.giveCommand(ShipCommand.ACCELERATE, null, 0);
            moveText = "前进";
        } else {
            // 后退
            moveDirection = ship.getFacing() + 180f;
            speedLimit = -MAX_BACKWARD_SPEED; // 注意这里是负值
            ship.giveCommand(ShipCommand.ACCELERATE_BACKWARDS, null, 0);
            moveText = "后退";
        }

        // 检查速度是否达到阈值
        boolean canMove = (throttle > 0 && speedInForwardDirection < speedLimit) ||
                (throttle < 0 && speedInForwardDirection > speedLimit);

        if (canMove) {
            float amount_force = ACCELERATION * Math.abs(throttle) * amount;
            // 均分到每一秒的加速度，确保不同帧率下舰船加速一致
            applyForce(ship, moveDirection, amount_force);

            /*
            if (timer.isTargetReached(engine, 1f)) {
                Vector2f offset = new Vector2f(ship.getLocation());
                offset.x += 100;
                offset.y += 100;

                engine.addFloatingText(offset, moveText + "开度" + (Math.abs(throttle) * 100) + "%" +
                                "航向角" + ship.getFacing() + "速度:" + speedInForwardDirection,
                        20f, Color.white, ship, 0.1f, 1f);

            }

             */
        }
    }

    // 可控制开度的转向方法
    // direction: -1到1，负值表示左转，正值表示右转
    public void turn(float direction, float amount) {
        // 限制输入范围在[-1,1]
        direction = Math.max(-1f, Math.min(1f, direction));

        // 获取当前角速度
        float currentAngularVelocity = ship.getAngularVelocity();

        // 计算目标角速度变化
        float targetAngularChange = TURN_ACCELERATION * direction * amount;

        // 计算新的角速度
        float newAngularVelocity = currentAngularVelocity + targetAngularChange;

        // 限制角速度在最大范围内
        if (Math.abs(newAngularVelocity) > MAX_TURN_SPEED) {
            newAngularVelocity = Math.signum(newAngularVelocity) * MAX_TURN_SPEED;
        }

        // 设置新的角速度
        ship.setAngularVelocity(newAngularVelocity);

        /*
        // 显示转向信息
        if (timer.isTargetReached(engine, 1f)) {
            Vector2f offset = new Vector2f(ship.getLocation());
            offset.x += 100;
            offset.y += 100;
            String turnDirectionText = direction > 0 ? "右转" : "左转";
            engine.addFloatingText(offset, turnDirectionText + "开度" + (Math.abs(direction) * 100) + "%",
                    20f, Color.white, ship, 0.1f, 1f);
        }

         */
    }

    // 可控制开度的平移方法
    // throttle: 开度，正值表示向右平移，负值表示向左平移，范围[-1,1]
    // amount: 时间因子，用于确保不同帧率下加速度一致
    public void strafe(float throttle, float amount) {
        // 限制开度范围在[-1,1]
        throttle = Math.max(-1f, Math.min(1f, throttle));

        // 如果开度为0，不执行任何操作
        if (throttle == 0) return;

        Vector2f velocity = new Vector2f(ship.getVelocity());

        // 获取船头方向的角度（以弧度表示）
        float facingRad = (float)Math.toRadians(ship.getFacing());

        // 计算侧向方向向量（船头方向旋转90度）
        // 向右平移：船头方向+90度
        // 向左平移：船头方向-90度
        Vector2f strafeDir = new Vector2f(
                (float)Math.cos(facingRad + Math.PI/2),  // x分量
                (float)Math.sin(facingRad + Math.PI/2)   // y分量
        );

        // 计算速度在侧向方向上的投影
        float speedInStrafeDirection = Vector2f.dot(velocity, strafeDir);

        // 根据开度正负确定平移方向和速度限制
        float strafeDirection;
        float speedLimit;
        String strafeText;

        if (throttle > 0) {
            // 向右平移
            strafeDirection = ship.getFacing() + 90f;
            speedLimit = MAX_STRAFE_SPEED;
            strafeText = "右平移";
        } else {
            // 向左平移
            strafeDirection = ship.getFacing() - 90f;
            speedLimit = -MAX_STRAFE_SPEED; // 注意这里是负值
            strafeText = "左平移";
        }

        // 检查速度是否达到阈值
        boolean canStrafe = (throttle > 0 && speedInStrafeDirection < speedLimit) ||
                (throttle < 0 && speedInStrafeDirection > speedLimit);

        if (canStrafe) {
            float amount_force = STRAFE_ACCELERATION * Math.abs(throttle) * amount;
            // 施加平移力
            applyForce(ship, strafeDirection, amount_force);
            /*
            // 显示平移信息
            if (timer.isTargetReached(engine, 1f)) {
                Vector2f offset = new Vector2f(ship.getLocation());
                offset.x += 100;
                offset.y += 100;
                engine.addFloatingText(offset, strafeText + "开度" + (Math.abs(throttle) * 100) + "%" +
                                "侧向速度:" + speedInStrafeDirection,
                        20f, Color.white, ship, 0.1f, 1f);
            }

             */
        }
    }
}