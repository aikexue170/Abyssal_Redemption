package data.scripts.utils;

import com.fs.starfarer.api.Global;
import com.fs.starfarer.api.combat.CombatEngineAPI;
import com.fs.starfarer.api.combat.ShipAPI;
import org.lwjgl.util.vector.Vector2f;

import java.awt.*;

public class ARR_GhostUtil {

    public static void AddStaticGhost(ShipAPI ship, Color color, float in, float dur, float out,
                                      boolean additive, boolean combineColor, boolean aboveShip) {

        // 检查 ship 是否为 null
        if (ship == null) {
            System.err.println("Error: Ship is null when generating ghost");
            return;
        }

        // 获取单例的计时器管理器
        ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();

        // 获取战斗引擎
        CombatEngineAPI engine = Global.getCombatEngine();

        // 获取或创建该船的计时器
        ARR_Timer timer = timerManager.getTimerForEntity(ship);

        // 获取当前时间
        float time = timer.getTotalTime();

        // 获取飞船的速度
        Vector2f velocity = new Vector2f(ship.getVelocity());

        // 计算速度的长度（即速度大小）
        float speed = velocity.length();

        // 如果速度为0，直接返回，避免除以0的情况
        if (speed == 0) {
            return;
        }

        // 计算每个拖尾片段的间距
        // 你可以根据需要调整这个值，例如每秒生成多少个拖尾片段
        float segmentSpacing = speed / 3f; // 每秒生成10个拖尾片段

        // 获取当前速度的方向并归一化
        Vector2f normalizedVelocity = velocity.normalise(null);

        // 创建一个新的向量来存储修改后的速度
        Vector2f modifiedVelocity = new Vector2f(normalizedVelocity);

        // 计算拖尾的偏移量
        for (int i = 1; i <= 3; i++) {
            float output = i * segmentSpacing; // 根据速度调整拖尾间距
            modifiedVelocity.scale(output).negate(); // 计算偏移量

            if (timer.isTargetReached(engine, 0.1f)) {
                // 添加拖尾效果
                ship.addAfterimage(color,
                        modifiedVelocity.x, // 使用带有偏移量的 x 坐标
                        modifiedVelocity.y, // 使用带有偏移量的 y 坐标
                        0, // 不需要额外的速度分量
                        0, // 不需要额外的速度分量
                        0.1f * i, // 逐渐增加透明度
                        in,
                        dur,
                        out,
                        additive,
                        combineColor,
                        aboveShip);
            }

            // 重置 modifiedVelocity 以便下次循环使用
            modifiedVelocity.set(normalizedVelocity);
        }
    }

    public static void AddFluentGhost(ShipAPI ship, Color color, float in, float dur, float out,
                                      boolean additive, boolean combineColor, boolean aboveShip) {

        // 检查 ship 是否为 null
        if (ship == null) {
            System.err.println("Error: Ship is null in PRI_addAfterImage");
            return;
        }

        // 获取单例的计时器管理器
        ARR_EntityTimerManager timerManager = ARR_EntityTimerManager.getInstance();

        // 获取战斗引擎
        CombatEngineAPI engine = Global.getCombatEngine();

        // 获取或创建该船的计时器
        ARR_Timer timer = timerManager.getTimerForEntity(ship);

        // 获取当前时间
        float time = timer.getTotalTime();

        // 获取飞船的速度
        Vector2f velocity = new Vector2f(ship.getVelocity());

        // 计算速度的长度（即速度大小）
        float speed = velocity.length();

        // 获取当前速度的方向并归一化
        Vector2f normalizedVelocity = velocity.normalise(null);

        // 计算拖尾片段的运动方向（速度的反方向）
        Vector2f ghostDirection = new Vector2f(normalizedVelocity);
        ghostDirection.negate(); // 反向

        // 计算拖尾片段的运动速度（可以根据需要调整）
        float ghostSpeed = speed * 1.5f; // 拖尾片段的速度为飞船速度的 150%

        // 检查是否满足生成拖尾的条件
        if (timer.isTargetReached(engine, 0.1f)) {
            // 添加拖尾效果
            for (int i = 1; i <= 3; i++) {
                // 添加拖尾效果
                ship.addAfterimage(
                        color, // 拖尾的颜色
                        0, // 拖尾的 x 坐标偏移
                        0, // 拖尾的 y 坐标偏移
                        ghostDirection.x * ghostSpeed, // 拖尾的 x 方向速度
                        ghostDirection.y * ghostSpeed, // 拖尾的 y 方向速度
                        0.1f * i, // 逐渐增加透明度
                        in, // 拖尾的淡入时间
                        dur, // 拖尾的持续时间
                        out, // 拖尾的淡出时间
                        additive,
                        combineColor,
                        aboveShip);
            }
        }
    }


}