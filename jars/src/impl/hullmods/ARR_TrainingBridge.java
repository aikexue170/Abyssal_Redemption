package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.BaseHullMod;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.network_training.ARR_StateSender;

public class ARR_TrainingBridge extends BaseHullMod {

    private static final float SEND_INTERVAL = 0.1f;
    private static float lastSendTime = -999f;
    private static ARR_StateSender sender = null;

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null || !ship.isAlive()) return;

        CombatEngineAPI engine = Global.getCombatEngine();
        if (engine == null) return;

        if (sender == null) {
            sender = new ARR_StateSender();
            sender.connect();
        }

        float now = engine.getTotalElapsedTime(false);
        if (now - lastSendTime >= SEND_INTERVAL) {
            lastSendTime = now;

            sender.sendAllStates();

            receiveActions();
        }
    }

    private void receiveActions() {
        while (sender.hasMessage()) {
            String msg = sender.getMessage();
            if (msg == null || msg.equals("END")) continue;

            String[] parts = msg.split(",");
            if (parts.length >= 4) {
                System.out.println("[训练桥接] 收到动作: " + msg);
            }
        }
    }
}
