# network_training — 环境参数采样与拟合

对游戏内舰船施加标准动作，采集状态时间序列，用最小二乘法拟合运动参数
（加速/减速阻尼、旋转、平移），为强化学习环境建模提供真实参数。

## 文件

| 文件 | 说明 |
|------|------|
| `protocol.py` | Java ↔ Python 行文本协议编解码（端口 65432） |
| `training_server.py` | TCP 服务器，接收状态帧、下发动作/复位指令 |
| `run_sampling.py` | 采样主程序：动作矩阵 × 每组 3 次，自动复位归位 |
| `analyze_fit.py` | 最小二乘拟合 + matplotlib 可视化 |
| `evaluate_model.py` | 复合运动评估：游戏真实轨迹 vs 拟合模型仿真轨迹对比 |

## 协议

    Java -> Python:  STATE,<simTime>,<shipId>,<x>,<y>,<vx>,<vy>,<facing>,<angVel>  ... END
    Python -> Java:  ACT,<shipId>,<move>,<turn>,<strafe> / RESET,<shipId|ALL>  ... END

- 三轴开度范围 [-1,1]，由游戏内 `ShipControlSystem` 施加
- `RESET` 把舰船传送回出生姿态并清零速度/角速度（每轮测试前自动调用）

## 使用方法

1. 构建 jar 并启动游戏，进入战斗，给被测舰船装上 **训练桥接器**（ARR_TrainingBridge）
2. `python run_sampling.py` —— 自动跑完动作矩阵，样本存到 `output/samples_*.csv`
3. `python analyze_fit.py` —— 拟合并输出 `output/params.json` 与 `fit_*.png`
4. `python evaluate_model.py` —— 播放 30s 复合运动并用拟合参数离线仿真（dt=0.05），
   输出轨迹对比图与位置误差统计（参考：平均 ≈17-21 su，轨迹长 ≈940 su）

## 拟合模型（分段恒定加速度，一切从简）

    油门期:  dv/dt = a*u        恒定加速度，与 v 无关，分方向拟合 a+ / a-
            越过施力上限 force_limit 后只剩很小的恒定爬行 creep
    上限:    v 最终钳制在 [v_cap_neg, v_cap_pos]
    滑行期:  dv/dt ≈ 0          阻力近似为零（coast_decel_measured 仅为参考值）

实测参考（2026-09-08 采样，临渊）：
    move:   a± ≈ 13.8 su/s²,  施力上限 +40/-20, 最终上限 ≈ +49/-43（爬行 ≈ ±1）
    strafe: a± ≈ 7.56 su/s², 上限 ±15
    turn:   a± ≈ 19.0 °/s²,  上限 ±20 °/s（滑行有 ≈1 °/s² 的恒定角阻尼，模型忽略）

## 已知坑（务必注意）

- **游戏直接上报的速度与位置存在时序不同步**，拟合必须使用位置/朝向差分
  得到的速度（analyze_fit.py 已内置，对比图见 velocity_source_compare.png）
- 满开度动作会迅速顶到速度硬上限；采样矩阵中已包含 ±0.2 小开度动作，
  且动作持续时间自适应（检测到平台期才松开），保证完整轨迹被记录
- `run_sampling.py --drag` 可选：全程滑行采样（松开后记录到完全停住），
  用于需要精细刻画阻力曲线时再启用；当前模型按零阻力处理，不需要运行
