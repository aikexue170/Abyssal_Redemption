package impl.combat.system;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.CombatEntityAPI;
import com.fs.starfarer.api.combat.MutableShipStatsAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import com.fs.starfarer.api.impl.combat.BaseShipSystemScript;
import data.scripts.utils.*;
import org.lazywizard.lazylib.combat.CombatUtils;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class DimensionDrivenField extends BaseShipSystemScript {

	// 偏导立场实例
	private ARR_DeflectionFieldUtil deflectionField;

	// 实体列表
	protected List<CombatEntityAPI> entities = new ArrayList<>();

	// 构造函数
	public DimensionDrivenField() {
		// 初始化偏导立场
		deflectionField = new ARR_DeflectionFieldUtil();
	}

	@Override
	public void apply(MutableShipStatsAPI stats, String id, State state, float effectLevel) {
		CombatEngineAPI engine = Global.getCombatEngine();
		ShipAPI ship = (ShipAPI) stats.getEntity();


		engine.getShips();
		// null检测
		if (ship == null) {
			System.out.println("【DimensionDrivenField】 ship is null");
			return;
		}

		// 获取飞船船尾处圆环的位置
		Vector2f shipLocation = ARR_LocationUtil.offsetPositionOnShip(ship, -80);

		// 为飞船本身添加拖尾效果
		ARR_GhostUtil.AddFluentGhost(ship, new Color(255, 51, 170, 170), 0.1f, 0.02f, 0.2f, false, true, false);
		applyJitterEffect(ship);

		// 更新偏导立场的中心位置
		deflectionField.setFieldCenter(shipLocation);

		// 获取周围650格范围内的所有实体
		entities = CombatUtils.getEntitiesWithinRange(shipLocation, 650);

		// 应用偏导立场效果
		applyDeflectionForceToEntities(entities, ship);

		// 对加速度/速度的更改
		if (state == State.OUT) {
			stats.getMaxSpeed().unmodify(id);
			stats.getAcceleration().unmodify(id);
			Vector2f vector = ship.getVelocity().normalise(null);
			vector.scale(ship.getMaxSpeed());
			Vector2f.add(new Vector2f(0, 0), vector, ship.getVelocity());
			ARR_TemporalShellUtil.unapplyTemporalShell(ship);
		} else {
			stats.getMaxSpeed().modifyFlat(id, 400f * effectLevel);
			stats.getAcceleration().modifyFlat(id, 300f * effectLevel);
			// 为飞船添加扭曲效果
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

	// 为进入偏导立场的实体施加偏转力
	private void applyDeflectionForceToEntities(List<CombatEntityAPI> entities, ShipAPI ship) {
		Vector2f shipFacingDirection = deflectionField.getShipFacingDirection(ship.getFacing());
		for (CombatEntityAPI entity : entities) {
			// 获取实体的位置和速度
			Vector2f entityPosition = entity.getLocation();
			Vector2f entityVelocity = entity.getVelocity();

			// 为实体施加偏转力
			deflectionField.applyDeflectionForce(entityPosition, entityVelocity, entity, shipFacingDirection);
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