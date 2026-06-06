package data.scripts.utils;

import com.fs.starfarer.api.combat.CombatEntityAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import org.lazywizard.lazylib.combat.CombatUtils;
import org.lwjgl.util.vector.Vector2f;

public class ARR_DeflectionFieldUtil {

    public static final int GRID_SIZE = 50;  // 网格大小
    public static final float RADIUS = 600.0f;  // 偏导立场的半径
    private Vector2f FIELD_CENTER;  // 偏导立场的中心（动态设置）

    // 构造函数
    public ARR_DeflectionFieldUtil() {
        // 不需要初始化流场数组，因为所有方向都是相同的
    }

    // 设置偏导立场的中心位置
    public void setFieldCenter(Vector2f center) {
        this.FIELD_CENTER = center;
    }

    // 获取舰船的朝向方向向量
    public Vector2f getShipFacingDirection(float shipFacingAngle) {
        float angleInRadians = (float) Math.toRadians(shipFacingAngle);
        return new Vector2f((float) Math.cos(angleInRadians), (float) Math.sin(angleInRadians));
    }

    // 为实体施加偏转力并更新其朝向
    public void applyDeflectionForce(Vector2f entityPosition, Vector2f entityVelocity, CombatEntityAPI entity, Vector2f shipFacingDirection) {
        if (entity instanceof ShipAPI && ((ShipAPI)entity).getHullSize() != ShipAPI.HullSize.FIGHTER) {
            return;  // 如果是飞船，直接返回，不施加力
        }

        // 计算实体与偏导立场中心的距离
        Vector2f relativePosition = new Vector2f();
        Vector2f.sub(entityPosition, FIELD_CENTER, relativePosition);
        float distanceToShip = relativePosition.length();

        // 只对在作用范围内的实体施加偏转力
        if (distanceToShip > RADIUS) {
            return;
        }

        // 设置偏转方向为舰船的反向
        Vector2f deflectionDirection = new Vector2f(-shipFacingDirection.x, -shipFacingDirection.y);
        deflectionDirection.normalise();

        // 计算力的大小
        float forceMagnitude = calculateForceMagnitude(distanceToShip);

        // 如果不是激光之类的，施加力
        if(entity.getMass() != 0f){
            CombatUtils.applyForce(entity, deflectionDirection, forceMagnitude);
            // 更新实体的朝向为当前速度方向
            updateEntityFacing(entity, entityVelocity);
        }
    }

    // 力计算函数
    private float calculateForceMagnitude(float distance) {
        float maxForce = 50f;
        float minForce = 15f;
        return maxForce * (minForce / (distance + minForce));
    }

    // 计算相对位置向量
    private Vector2f getRelativePosition(Vector2f position) {
        Vector2f relativePosition = new Vector2f();
        Vector2f.sub(position, FIELD_CENTER, relativePosition);
        return relativePosition;
    }

    // 更新实体的朝向为当前速度方向
    private void updateEntityFacing(CombatEntityAPI entity, Vector2f velocity) {
        // 检查速度是否为零向量
        if (velocity.lengthSquared() <= 0.0001f) {
            return;  // 如果速度为零，不更新朝向
        }

        // 计算速度方向的角度（以弧度为单位）
        float angleInRadians = (float) Math.atan2(velocity.y, velocity.x);

        // 将角度从弧度转换为度
        float angleInDegrees = (float) Math.toDegrees(angleInRadians);

        // 设置实体的朝向
        entity.setFacing(angleInDegrees);
    }
}