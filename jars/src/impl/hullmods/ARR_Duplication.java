package impl.hullmods;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.BaseHullMod;
import com.fs.starfarer.api.combat.CollisionClass;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.ARR_SpawnManager;
import data.scripts.utils.ARR_Timer;
import org.lwjgl.util.vector.Vector2f;

import java.util.ArrayList;
import java.util.List;

/**
 * 多维并行器 — 分三阶段执行：
 * <ol>
 *   <li>进入战斗 2 秒：生成 N 艘同配置舰船，立即关闭碰撞</li>
 *   <li>再等 5 秒：将所有复制体统一排列至指定位置和朝向</li>
 *   <li>完成，不再动作</li>
 * </ol>
 * 使用静态共享计时器 + 每帧防重入，整场战斗只运行一次。
 */
public class ARR_Duplication extends BaseHullMod {

    /** 生成数量 */
    private static final int SPAWN_COUNT = 199;
    /** 第一阶段延迟 — 生成（秒） */
    private static final float PHASE_SPAWN_DELAY = 2f;
    /** 第二阶段延迟 — 统一排列（秒），从生成完成后算起 */
    private static final float PHASE_POSITION_DELAY = 5f;
    /** 统一排列位置 */
    private static final float FORMATION_X = 0f;
    private static final float FORMATION_Y = 1500f;
    /** 统一朝向 */
    private static final float FORMATION_FACING = 90f;

    private static final ARR_SpawnManager spawnManager = ARR_SpawnManager.getInstance();

    /** 静态共享计时器 */
    private static final ARR_Timer sharedTimer = new ARR_Timer();
    /** 复制体列表 */
    private static final List<ShipAPI> spawnedShips = new ArrayList<>();
    /** 当前阶段：0=等待生成, 1=等待定位, 2=已完成 */
    private static int phase = 0;

    /** 每帧防重入 — 避免 removeMod 未生效时多个实例重复驱动状态机 */
    private static float lastFrameTimestamp = -1f;

    @Override
    public void advanceInCombat(ShipAPI ship, float amount) {
        if (ship == null || !ship.isAlive()) return;
        if (phase >= 2) return;

        CombatEngineAPI engine = Global.getCombatEngine();
        if (engine == null) return;

        // 每帧只允许一个实例推进逻辑，防止计时器倍速
        float now = engine.getTotalElapsedTime(false);
        if (now == lastFrameTimestamp) return;
        lastFrameTimestamp = now;

        switch (phase) {
            case 0:
                if (!sharedTimer.isTargetReached(engine, PHASE_SPAWN_DELAY)) return;

                List<ShipAPI> spawned = spawnManager.spawnShip(
                        0,
                        ship.getVariant(),
                        ship.getLocation(),
                        ship.getVelocity(),
                        ship.getFacing(),
                        ship.getAngularVelocity(),
                        false,
                        SPAWN_COUNT);

                for (ShipAPI s : spawned) {
                    s.getVariant().removeMod("ARR_Duplication");
                    s.setCollisionClass(CollisionClass.NONE);
                    spawnedShips.add(s);
                }
                phase = 1;
                break;

            case 1:
                if (!sharedTimer.isTargetReached(engine, PHASE_POSITION_DELAY)) return;

                for (ShipAPI s : spawnedShips) {
                    if (s.isAlive()) {
                        s.getLocation().set(FORMATION_X, FORMATION_Y);
                        s.setFacing(FORMATION_FACING);
                        s.getVelocity().set(0, 0);
                        s.setAngularVelocity(0);
                    }
                }
                phase = 2;
                break;
        }
    }
}