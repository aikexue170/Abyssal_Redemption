package data.scripts.utils;

import com.fs.starfarer.api.combat.ShipAPI;

import java.util.ArrayList;
import java.util.List;

import com.fs.starfarer.api.combat.ShipVariantAPI;
import data.scripts.utils.ARR_Spawn;
import org.lwjgl.util.vector.Vector2f;

public class ARR_SpawnManager {

    // 单例模式：确保只有一个管理器实例
    private static final ARR_SpawnManager instance = new ARR_SpawnManager();

    // 私有构造函数，防止外部实例化
    private ARR_SpawnManager() {}

    // 获取单例实例
    public static ARR_SpawnManager getInstance() {
        return instance;
    }

    // 在指定位置生成指定数量的舰船
    /**
     * 生成舰船，默认满结构值投入战斗。
     *
     * @param owner    所属方，0 = 玩家方，1 = 敌方
     * @param variant  舰船变体（包含装配方案）
     * @param location 生成位置（世界坐标）
     * @param velocity 初始速度向量
     * @param facing   初始朝向角度（度）
     * @param angular  初始角速度
     * @param isAlly   是否视为友军
     * @param number   生成数量
     */
    public List<ShipAPI> spawnShip(int owner, ShipVariantAPI variant, Vector2f location, Vector2f velocity, float facing, float angular, boolean isAlly, int number){
        List<ShipAPI> shipList = new ArrayList<>();

        for(int i = 0; i < number; i++){
            ShipAPI ship = ARR_Spawn.spawnShip(owner, variant, location, velocity, facing, angular, isAlly);
            shipList.add(ship);
        }

        return shipList;
    }
}
