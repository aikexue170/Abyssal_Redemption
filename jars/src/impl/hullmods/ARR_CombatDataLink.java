package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.BaseHullMod;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.ARR_EntityTimerManager;
import data.scripts.utils.ARR_ShipData;
import data.scripts.utils.ARR_ShipDataManager;
import data.scripts.utils.ARR_Timer;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;

public class ARR_CombatDataLink extends BaseHullMod {

    private static final ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();
    private static final ARR_ShipDataManager dataManager = ARR_ShipDataManager.getInstance();

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null || !ship.isAlive()) return;

        dataManager.register(ship);

        CombatEngineAPI engine = Global.getCombatEngine();
        if (engine == null) return;

        ARR_ShipData data = dataManager.get(ship);
        data.update(ship.getLocation(), ship.getVelocity(), ship.getFacing(), ship.getAngularVelocity());

//        ARR_Timer timer = timerManager.getTimerForEntity(ship);
//
//        if (timerManager.isTargetReachedForEntity(ship, engine, 2f)) {
//            timerManager.resetTimerForEntity(ship);
//
//            Vector2f loc = ship.getLocation();
//            engine.addFloatingText(new Vector2f(loc.x - 100, loc.y),
//                    "已注册: " + ship.getId(), 20f, Color.green, ship, 0.0001f, 0.0001f);
//            engine.addFloatingText(new Vector2f(loc.x - 100, loc.y + 40),
//                    "注册表总数: " + dataManager.size(), 20f, Color.cyan, ship, 0.0001f, 0.0001f);
//        }
    }
}
