package data.scripts.utils;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.ShipAPI;
import com.fs.starfarer.api.combat.ShipVariantAPI;
import com.fs.starfarer.api.fleet.FleetMemberAPI;
import com.fs.starfarer.api.fleet.FleetMemberType;
import org.lwjgl.util.vector.Vector2f;

public class ARR_Spawn
{
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
     */
    public static ShipAPI spawnShip(int owner, ShipVariantAPI variant, Vector2f location, Vector2f velocity, float facing, float angular, boolean isAlly)
    {
        boolean suppress = Global.getCombatEngine().getFleetManager(owner).isSuppressDeploymentMessages();
        Global.getCombatEngine().getFleetManager(owner).setSuppressDeploymentMessages(true);
        FleetMemberAPI member = createMember(variant);
        member.setOwner(owner);
        member.setAlly(isAlly);
        ShipAPI newShip = Global.getCombatEngine().getFleetManager(owner).spawnFleetMember(member, location, facing, 0f);

        newShip.setAlly(isAlly);
        newShip.setOwner(owner);
        newShip.setCurrentCR(0.7f);
        newShip.getVelocity().set(velocity);
        newShip.setAngularVelocity(angular);

        Global.getCombatEngine().getFleetManager(owner).setSuppressDeploymentMessages(suppress);
        return newShip;
    }

    /**
     * 生成指定结构值的残血舰船。
     *
     * @param owner     所属方，0 = 玩家方，1 = 敌方
     * @param variant   舰船变体（包含装配方案）
     * @param location  生成位置（世界坐标）
     * @param velocity  初始速度向量
     * @param hitpoints 初始结构值（用于生成残血/练刀靶子）
     * @param facing    初始朝向角度（度）
     * @param angular   初始角速度
     * @param isAlly    是否视为友军
     */
    public static ShipAPI spawnShip(int owner, ShipVariantAPI variant, Vector2f location, Vector2f velocity, float hitpoints, float facing, float angular, boolean isAlly)
    {
        ShipAPI newShip = spawnShip(owner, variant, location, velocity, facing, angular, isAlly);
        newShip.setHitpoints(hitpoints);

        return newShip;
    }

    /**
     * 根据变体创建舰队成员，装填满编船员并设置 70% 战备值。
     *
     * @param variant 舰船变体（包含装配方案）
     * @return 已配置好船员和战备的 FleetMemberAPI
     */
    public static FleetMemberAPI createMember(ShipVariantAPI variant)
    {
        FleetMemberAPI member = Global.getFactory().createFleetMember(FleetMemberType.SHIP, variant);
        member.getRepairTracker().setCR(0.7f);
        member.getCrewComposition().addCrew(member.getHullSpec().getMaxCrew());
        member.getRepairTracker().setCrashMothballed(false);
        member.getRepairTracker().setMothballed(false);
        return member;
    }
}
