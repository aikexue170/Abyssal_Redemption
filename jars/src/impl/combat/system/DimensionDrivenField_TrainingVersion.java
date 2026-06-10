package impl.combat.system;

import com.fs.starfarer.api.combat.CollisionClass;
import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.MutableShipStatsAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import com.fs.starfarer.api.impl.combat.BaseShipSystemScript;
import org.lwjgl.util.vector.Vector2f;

public class DimensionDrivenField_TrainingVersion extends BaseShipSystemScript {

	@Override
	public void apply(MutableShipStatsAPI stats, String id, State state, float effectLevel) {
		CombatEngineAPI engine = Global.getCombatEngine();
		ShipAPI ship = (ShipAPI) stats.getEntity();

		if (ship == null) return;

		ship.setCollisionClass(CollisionClass.NONE);

		if (state == State.OUT) {
			stats.getMaxSpeed().unmodify(id);
			stats.getAcceleration().unmodify(id);
			Vector2f velocity = ship.getVelocity();
			float speed = velocity.length();

			if (speed > 0.1f) {
				Vector2f vector = new Vector2f(velocity.x / speed, velocity.y / speed);
				vector.scale(ship.getMaxSpeed());
				ship.getVelocity().set(vector);
			}
		} else {
			stats.getMaxSpeed().modifyFlat(id, 400f * effectLevel);
			stats.getAcceleration().modifyFlat(id, 300f * effectLevel);
		}
	}

	@Override
	public void unapply(MutableShipStatsAPI stats, String id) {
		stats.getMaxSpeed().unmodify(id);
		stats.getAcceleration().unmodify(id);
	}
}