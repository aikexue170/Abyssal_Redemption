package data.scripts.utils;

import com.fs.starfarer.api.combat.ShipAPI;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

public class ARR_ShipDataManager {

    private static final ARR_ShipDataManager instance = new ARR_ShipDataManager();

    /** LinkedHashMap 保证注册顺序即迭代顺序，与通讯序号一致 */
    private final Map<String, ARR_ShipData> shipDataMap = new LinkedHashMap<>();
    private int nextIndex = 0;

    private ARR_ShipDataManager() {}

    public static ARR_ShipDataManager getInstance() {
        return instance;
    }

    public synchronized void register(ShipAPI ship) {
        if (ship == null) return;
        String id = ship.getId();
        if (!shipDataMap.containsKey(id)) {
            ARR_ShipData data = new ARR_ShipData(ship);
            data.index = nextIndex++;
            shipDataMap.put(id, data);
        }
    }

    public synchronized ARR_ShipData get(String shipId) {
        return shipDataMap.get(shipId);
    }

    public synchronized ARR_ShipData get(ShipAPI ship) {
        if (ship == null) return null;
        return shipDataMap.get(ship.getId());
    }

    public synchronized void unregister(String shipId) {
        shipDataMap.remove(shipId);
    }

    public synchronized void unregister(ShipAPI ship) {
        if (ship != null) {
            shipDataMap.remove(ship.getId());
        }
    }

    public synchronized Collection<ARR_ShipData> getAll() {
        return shipDataMap.values();
    }

    public synchronized int size() {
        return shipDataMap.size();
    }

    public synchronized void clear() {
        shipDataMap.clear();
    }
}
