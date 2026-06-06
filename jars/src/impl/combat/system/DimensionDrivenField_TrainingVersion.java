package impl.combat.system;

import com.fs.starfarer.api.combat.CollisionClass;
import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.MutableShipStatsAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import com.fs.starfarer.api.impl.combat.BaseShipSystemScript;
import data.scripts.utils.*;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;

public class DimensionDrivenField_TrainingVersion extends BaseShipSystemScript {

	@Override
	public void apply(MutableShipStatsAPI stats, String id, State state, float effectLevel) {
		CombatEngineAPI engine = Global.getCombatEngine();
		ShipAPI ship = (ShipAPI) stats.getEntity();

		if (ship == null) return;

		ship.setCollisionClass(CollisionClass.NONE);

		// 获取飞船船尾处圆环的位置
		Vector2f shipLocation = ARR_LocationUtil.offsetPositionOnShip(ship, -80);

		// 为飞船本身添加拖尾效果
		ARR_GhostUtil.AddFluentGhost(ship, new Color(255, 51, 170, 170), 0.1f, 0.02f, 0.2f, false, true, false);
		applyJitterEffect(ship);

		// 对加速度/速度的更改
		if (state == State.OUT) {
			stats.getMaxSpeed().unmodify(id);
			stats.getAcceleration().unmodify(id);
			// 安全的归一化，避免零向量
			Vector2f velocity = ship.getVelocity();
			float speed = velocity.length();

			if (speed > 0.1f) { // 只有有速度时才归一化
				Vector2f vector = new Vector2f(velocity.x / speed, velocity.y / speed);
				vector.scale(ship.getMaxSpeed());
				ship.getVelocity().set(vector);
			}
			// 如果速度很小，保持原样或设置为零
			ARR_TemporalShellUtil.unapplyTemporalShell(ship);
		}else {
			stats.getMaxSpeed().modifyFlat(id, 400f * effectLevel);
			stats.getAcceleration().modifyFlat(id, 300f * effectLevel);
			ARR_DistortionUtil.ringDistortion(shipLocation, ship.getVelocity(), 30f, 0f, 0.5f, 0.2f);
			//添加时流效果
			ARR_TemporalShellUtil.applyTemporalShell(ship,
					13f,
					false,
					new Color(255, 0, 100, 170),
					new Color(205, 10, 250, 170)
			);
		}
	}
	private void applyJitterEffect(ShipAPI ship) {
		ship.setJitter(ship, new Color(255, 51, 170, 170), 0.3f, 4, 50f);
	}

	@Override
	public void unapply(MutableShipStatsAPI stats, String id) {
		stats.getMaxSpeed().unmodify(id);
		stats.getAcceleration().unmodify(id);
	}
}