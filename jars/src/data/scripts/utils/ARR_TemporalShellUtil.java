package data.scripts.utils;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.MutableShipStatsAPI;
import com.fs.starfarer.api.combat.ShipAPI;

import java.awt.*;

public class ARR_TemporalShellUtil {

    // 定义时流效果的状态
    private static final String TEMPORAL_SHELL_ACTIVE = "temporal_shell_active";

    // 定义默认的颜色抖动
    private static final Color DEFAULT_JITTER_COLOR = new Color(90, 165, 255, 55);
    private static final Color DEFAULT_JITTER_UNDER_COLOR = new Color(90, 165, 255, 155);

    /**
     * 应用时流效果，修改舰船的时间流逝倍率并处理视角转换
     *
     * @param ship           舰船对象
     * @param mult           时间流逝倍率
     * @param changeViewpoint 是否转换视角
     * @param jitter         引擎颜色抖动的颜色（可选）
     * @param jitterUnder    引擎底部颜色抖动的颜色（可选）
     */
    public static void applyTemporalShell(ShipAPI ship, float mult, boolean changeViewpoint, Color jitter, Color jitterUnder) {
        if (ship == null || ship.getMutableStats() == null) {
            return;  // 如果舰船或其统计信息为空，直接返回
        }

        // 获取舰船的统计信息
        MutableShipStatsAPI stats = ship.getMutableStats();
        String id = "temporal_shell_" + ship.getId();

        // 获取当前战斗引擎
        CombatEngineAPI engine = Global.getCombatEngine();

        // 检查时流效果是否已经启动
        boolean isActive = ship.getCustomData().containsKey(TEMPORAL_SHELL_ACTIVE);

        // 如果时流效果尚未启动，则初始化
        if (!isActive) {
            ship.getCustomData().put(TEMPORAL_SHELL_ACTIVE, true);
        }

        // 获取当前效果级别（假设效果级别为 1，因为我们不再使用 in/dur/out）
        float effectLevel = 1f;

        // 计算舰船的时间流逝倍率
        float shipTimeMult = 1f + (mult - 1f) * effectLevel;

        // 设置颜色抖动（如果未提供自定义颜色，则使用默认颜色）
        Color jitterColor = jitter != null ? jitter : DEFAULT_JITTER_COLOR;
        Color jitterUnderColor = jitterUnder != null ? jitterUnder : DEFAULT_JITTER_UNDER_COLOR;

        // 根据 changeViewpoint 参数调整时间流逝倍率和视觉效果
        if (changeViewpoint) {
            // 玩家视角下，周围世界变慢，自己无变化
            if (ship == engine.getPlayerShip()) {
                // 修改全局时间流逝倍率
                engine.getTimeMult().modifyMult(id, 1f / shipTimeMult);
            } else {
                // 非玩家舰船，仅修改自身时间流逝倍率
                engine.getTimeMult().unmodify(id);
            }
        } else {
            // 玩家视角下，船只突然变快，周围世界不变
            if (ship == engine.getPlayerShip()) {
                // 不修改全局时间流逝倍率
                engine.getTimeMult().unmodify(id);
            } else {
                // 非玩家舰船，修改自身时间流逝倍率
                engine.getTimeMult().modifyMult(id, 1f / shipTimeMult);
            }
        }

        // 修改舰船的时间流逝倍率
        stats.getTimeMult().modifyMult(id, shipTimeMult);

        // 应用引擎颜色抖动效果
        ship.setJitter(ARR_TemporalShellUtil.class, jitterColor, effectLevel, 3, 0, 0);
        ship.setJitterUnder(ARR_TemporalShellUtil.class, jitterUnderColor, effectLevel, 25, 0f, 7f);

        // 修改引擎颜色
        ship.getEngineController().fadeToOtherColor(ARR_TemporalShellUtil.class, jitterColor, new Color(0, 0, 0, 0), effectLevel, 0.5f);
        ship.getEngineController().extendFlame(ARR_TemporalShellUtil.class, -0.25f, -0.25f, -0.25f);
    }

    public static void applyTemporalShell(ShipAPI ship, float mult, boolean changeViewpoint) {
        if (ship == null || ship.getMutableStats() == null) {
            return;  // 如果舰船或其统计信息为空，直接返回
        }

        // 获取舰船的统计信息
        MutableShipStatsAPI stats = ship.getMutableStats();
        String id = "temporal_shell_" + ship.getId();

        // 获取当前战斗引擎
        CombatEngineAPI engine = Global.getCombatEngine();

        // 检查时流效果是否已经启动
        boolean isActive = ship.getCustomData().containsKey(TEMPORAL_SHELL_ACTIVE);

        // 如果时流效果尚未启动，则初始化
        if (!isActive) {
            ship.getCustomData().put(TEMPORAL_SHELL_ACTIVE, true);
        }

        // 获取当前效果级别（假设效果级别为 1，因为我们不再使用 in/dur/out）
        float effectLevel = 1f;

        // 计算舰船的时间流逝倍率
        float shipTimeMult = 1f + (mult - 1f) * effectLevel;

        // 根据 changeViewpoint 参数调整时间流逝倍率和视觉效果
        if (changeViewpoint) {
            // 玩家视角下，周围世界变慢，自己无变化
            if (ship == engine.getPlayerShip()) {
                // 修改全局时间流逝倍率
                engine.getTimeMult().modifyMult(id, 1f / shipTimeMult);
            } else {
                // 非玩家舰船，仅修改自身时间流逝倍率
                engine.getTimeMult().unmodify(id);
            }
        } else {
            // 玩家视角下，船只突然变快，周围世界不变
            if (ship == engine.getPlayerShip()) {
                // 不修改全局时间流逝倍率
                engine.getTimeMult().unmodify(id);
            } else {
                // 非玩家舰船，修改自身时间流逝倍率
                engine.getTimeMult().modifyMult(id, 1f / shipTimeMult);
            }
        }

        // 修改舰船的时间流逝倍率
        stats.getTimeMult().modifyMult(id, shipTimeMult);
    }

    /**
     * 移除时流效果，恢复默认的时间流逝倍率和视觉效果
     *
     * @param ship 舰船对象
     */
    public static void unapplyTemporalShell(ShipAPI ship) {
        if (ship == null || ship.getMutableStats() == null) {
            return;  // 如果舰船或其统计信息为空，直接返回
        }

        // 获取舰船的统计信息
        MutableShipStatsAPI stats = ship.getMutableStats();
        String id = "temporal_shell_" + ship.getId();

        // 获取当前战斗引擎
        CombatEngineAPI engine = Global.getCombatEngine();

        // 移除时流效果
        engine.getTimeMult().unmodify(id);
        stats.getTimeMult().unmodify(id);

        // 移除引擎颜色抖动效果
        ship.setJitter(ARR_TemporalShellUtil.class, null, 0f, 0, 0, 0);
        ship.setJitterUnder(ARR_TemporalShellUtil.class, null, 0f, 0, 0);

        // 移除引擎颜色修改
        ship.getEngineController().fadeToOtherColor(ARR_TemporalShellUtil.class, null, null, 0f, 0f);
        ship.getEngineController().extendFlame(ARR_TemporalShellUtil.class, 0f, 0f, 0f);

        // 标记时流效果为未启动状态
        ship.getCustomData().remove(TEMPORAL_SHELL_ACTIVE);
    }
}