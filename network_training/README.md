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

## 协议

    Java -> Python:  STATE,<simTime>,<shipId>,<x>,<y>,<vx>,<vy>,<facing>,<angVel>  ... END
    Python -> Java:  ACT,<shipId>,<move>,<turn>,<strafe> / RESET,<shipId|ALL>  ... END

- 三轴开度范围 [-1,1]，由游戏内 `ShipControlSystem` 施加
- `RESET` 把舰船传送回出生姿态并清零速度/角速度（每轮测试前自动调用）

## 使用方法

1. 构建 jar 并启动游戏，进入战斗，给被测舰船装上 **训练桥接器**（ARR_TrainingBridge）
2. `python run_sampling.py` —— 自动跑完动作矩阵，样本存到 `output/samples_*.csv`
3. `python analyze_fit.py` —— 拟合并输出 `output/params.json` 与 `fit_*.png`

## 拟合模型

    dv/dt = a*u - b*v
    a: 单位开度加速度 (su/s² 或 度/s²)
    b: 阻尼系数 (1/s)，松开开度后 dv/dt = -b*v 即减速特性
    另报告观测速度硬上限（cap）

## 已知坑（务必注意）

- **游戏直接上报的速度与位置存在时序不同步**，拟合必须使用位置/朝向差分
  得到的速度（analyze_fit.py 已内置，对比图见 velocity_source_compare.png）
- 满开度动作会迅速顶到速度硬上限，线性区样本太少；采样矩阵中已包含
  ±0.2 小开度动作，拟合主要依赖这些未触顶的完整轨迹
