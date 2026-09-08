# rl/ —— GPU 张量化强化学习 (导航任务)

全链路不出 GPU 的单文件 PPO 实现。环境动力学来自 ../output/params.json
（analyze_fit.py 的实测拟合结果），与 evaluate_model.py 的参考仿真逐行对应。

## 文件

| 文件 | 职责 |
|------|------|
| tensor_env.py | 张量化导航环境：分段恒定加速度动力学的向量化实现；域随机化（每 env 参数 ±15%，reset 重采样）；STAGES 课程表（到点停下 → 收紧半径/速度 → 朝向对齐 ±20° → ±8°） |
| ppo.py | 单文件 PPO：rollout buffer 全程 CUDA tensor；GAE + clip + 截断自举。注意：buffer 必须存未 clamp 的原采样动作（logp 在 raw 上计算） |
| train_nav.py | 训练入口：滚动成功率 >70% 自动晋级，晋级时熵系数衰减（精密驻留要求策略收敛） |
| eval_policy.py | 加载存档，确定性策略跑一批回合，画轨迹图 + 成功率/终距/朝向误差统计 |
| diag_env.py | 环境诊断：手写 PD 控制器验证物理与成功判定（改环境后先跑它） |

## 用法

    conda activate starsector          # torch 2.12 + ROCm (RX 9070 XT)
    cd network_training/rl
    python train_nav.py --num-envs 2048 --num-steps 256 --iters 400 --run-name nav_vX
    python eval_policy.py --ckpt runs/nav_vX/latest.pt --episodes 16
    python diag_env.py                 # 环境改动后的快速体检

存档在 runs/<run_name>/（latest.pt / final.pt / config.json）。
吞吐参考：2048 env × 256 步，约 40-60 万步/秒（9070 XT）。

## 观测 / 动作约定

- 观测 8 维（船体系、归一化）：目标相对位置 ×2、船体系速度 vf/vs、角速度、朝向误差 sin/cos、距离
- 动作 3 维：[move, turn, strafe] ∈ [-1,1]，与 65432 协议 ACT 指令一一对应
- 回合 30s（600 步 × dt=0.05，与游戏仿真 tick 一致）

## 已踩过的坑（训练调参记录）

1. buffer 存 clamp 后动作 → logp 失配，approx-KL 虚高（~2），必须存 raw
2. 纯进展 shaping 会卡在距目标 ~110 su 的局部最优（≈ 满速刹停距离）：需要超时距离惩罚 + 近场高速惩罚
3. 熵系数过大且不衰退 → 油门噪声 σ≈0.6，1 秒精密驻留不可能成功；晋级时衰减熵系数
4. 首阶段课程要足够宽（半径 40、速度 <12、驻留 0.5s），否则成功信号迟迟无法建立
5. **成功大奖不能终止回合**：驻留价值流（0.2/步 × γ=0.99 ≈ 20）大于大奖（10）时，
   理性策略会故意停在成功门槛外蹭驻留奖励。改为一次性大奖（50）+ 回合继续
6. **必须要求"不转"**：只要求平动停稳时，策略学会保持 20°/s 满速自旋、
   纯靠 move/strafe 平移（直升机模式），±20° 窗口靠扫过碰运气能拿 65% 成功率，
   但 ±8° 窗口（0.8s 扫过 < 1s 驻留要求）物理上不可能 → 终段卡死 0%。
   修复：成功条件加 |ω|<3°/s，并加小额角速度惩罚（0.005×|ω|/20）

## 当前最好结果 (nav_v7)

600 轮 × 2048 env × 256 步 ≈ 3.1 亿步，约 12 分钟（9070 XT，~44 万步/秒）。
stage 3（半径 8 / 停速 3.5 / 停转 3°/s / 朝向 ±8° / 驻留 1s）：
训练成功率 86-93%，确定性评估 93.8%，平均终距 3.2 su、终速 0.52 su/s、朝向误差 4.5°。
存档：runs/nav_v7/final.pt
