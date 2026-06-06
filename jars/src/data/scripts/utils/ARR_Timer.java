package data.scripts.utils;

import com.fs.starfarer.api.combat.CombatEngineAPI;

public class ARR_Timer {

    private float totalTime;

    public ARR_Timer() {
        reset();
    }

    // 无属性计时器
    public float timer(CombatEngineAPI engine) {
        synchronized (this) {
            float frameTime = engine.getElapsedInLastFrame();
            totalTime += frameTime;
            return totalTime;
        }
    }

    // 重置计时器
    public void reset() {
        synchronized (this) {
            totalTime = 0f;
        }
    }

    // 目标计时器，到达目标计时返回一次true，重置计时器。未到达返回false
    public boolean isTargetReached(CombatEngineAPI engine, float timingTarget) {
        synchronized (this) {
            if (timer(engine) >= timingTarget) {
                reset();
                return true;
            } else {
                return false;
            }
        }
    }

    // 获取当前计时器的总时间
    public float getTotalTime() {
        synchronized (this) {
            return totalTime;
        }
    }

    // 暂停计时器
    public void pause() {
        // 可以在这里实现暂停逻辑，例如保存当前的frameTime
    }

    // 恢复计时器
    public void resume(CombatEngineAPI engine) {
        // 可以在这里实现恢复逻辑，例如从保存的frameTime继续计时
    }
}