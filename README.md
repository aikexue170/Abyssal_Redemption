# 临渊号曙光 (Abyssal Redemption)

远行星号 (Starsector) 0.98 强化学习舰船控制 mod。为 AI 自主作战提供训练与实验平台。

## 背景

一艘来自"深井"(Abyss) 的无人深空战列舰——**临渊**，搭载维度驱动场战术系统与 16 组扫描激光传感器阵列。舰船通过内置的神经网络桥接器与外部 Python 训练进程通信，可实现对舰船移动的实时远程控制与并行强化学习训练。

## 内容

### 舰船

| 名称 | ID | 类型 | 部署点 | 说明 |
|------|-----|------|--------|------|
| 临渊 | `ARR_Linyuan` | 主力舰 | 45 | 无护盾，维度驱动场，16 组扫描激光传感器 |
| 临渊虚影 | `ARR_Linyuan_ghost` | 主力舰 | 45 | 训练专用精简版：无碰撞箱、无武器、无引擎特效 |

### 战术系统

| 名称 | ID | 对应舰船 | 说明 |
|------|-----|----------|------|
| 维度驱动场 | `DimensionDrivenField` | 临渊 | 加速 + 偏导力场推飞弹 + 时流 + 扭曲特效 |
| 维度驱动场（训练） | `DimensionDrivenField_TrainingVersion` | 临渊虚影 | 精简版：仅关闭碰撞 + 加速，无渲染开销 |

### 武器

| 名称 | ID | 说明 |
|------|-----|------|
| 扫描激光 | `ARR_scanninglaser` | 内置传感器，探测前方 2000 格内的主力/巡洋舰距离并归一化输出 |

### 船插 (Hullmods)

| 名称 | ID | 说明 |
|------|-----|------|
| 舰船信息显示器 | `ARR_ShowShipState` | 调试用 — 在屏幕上显示舰船位置、速度、角速度、传感器数据 |
| 舰船移动控制器 | `ARR_ShipController` | 通过 Socket (8888) 接收远程 `move/turn/strafe` 指令控制舰船 |
| 战斗数据链 | `ARR_CombatDataLink` | 内建船插 — 自动将舰船注册到全局数据管理器 |
| 训练桥接器 | `ARR_TrainingBridge` | 每 0.1s 向 Python 训练服务器 (65432) 发送舰船状态并接收动作指令 |
| 多维并行器 | `ARR_Duplication` | 训练用 — 延迟生成 99 艘复制体，关闭碰撞后统一排列 |

### 星系

深井 (Abyss) 星系位于 `(7000, -10000)`，支持原版与 [Nexerelin](https://github.com/Histidine91/Nexerelin) 两种生成模式。

## 依赖

| 依赖 | 必需 | 说明 |
|------|------|------|
| [MagicLib](https://github.com/MagicLibStarsector/MagicLib) | ✅ | 基础工具库 |
| [LazyLib](https://github.com/LazyWizard/lazylib) | ✅ | 战斗工具库 |
| [GraphicsLib](https://github.com/darkrevenge/GraphicsLib) | 可选 | 环状扭曲特效（临渊维度驱动场），未安装则无特效 |

## 网络训练通信

```
┌──────────────────┐     Socket 65432     ┌──────────────────────┐
│  Starsector 战斗  │ ◄────────────────── ► │  Python 训练服务器    │
│  (ARR_TrainingBridge) │  状态/动作 传输    │  (server_receiver.py) │
│  (ARR_ShipController) │ ◄─── 8888 ────► │  (远程控制客户端)     │
└──────────────────┘                      └──────────────────────┘
```

- **65432 端口**：`ARR_TrainingBridge` 每 0.1 秒批量发送所有已注册舰船的状态（ID、位置、速度、角度、角速度），Python 端接收后返回动作指令
- **8888 端口**：`ARR_ShipController` 接收 `move:val,turn:val,strafe:val` 格式的远程控制指令
- Python 示例接收脚本：`network_training/server_receiver.py`
- 旧版 PPO/DQN 训练框架已归档至 `归档代码/old_ppo_framework/`

### 训练工作流

1. 装配舰船：临渊虚影 + 战斗数据链 + 训练桥接器 + 多维并行器
2. 启动 Python 训练服务器 (`server_receiver.py` 或自定义)
3. 进入战斗 → 2 秒后多维并行器生成 99 艘复制体
4. 7 秒后所有复制体统一排列至 `(0, 1500)` 朝向 90°
5. 训练桥接器开始向 Python 发送状态，接收控制指令

## 架构

```
jars/src/
├── data/scripts/
│   ├── ARR_ModPlugin.java              # Mod 入口
│   ├── ARR_WorldGenerate.java          # 星系生成
│   ├── utils/
│   │   ├── ARR_Spawn.java              # 舰船生成工具
│   │   ├── ARR_SpawnManager.java       # 批量生成管理器（单例）
│   │   ├── ARR_ShipData.java           # 舰船数据模型
│   │   ├── ARR_ShipDataManager.java    # 舰船数据注册表（单例）
│   │   ├── ARR_Timer.java              # 通用计时器
│   │   ├── ARR_EntityTimerManager.java # 实体计时器管理器（单例）
│   │   ├── ARR_GhostUtil.java          # 鬼影拖尾特效
│   │   ├── ARR_DeflectionFieldUtil.java# 偏导力场物理
│   │   ├── ARR_DistortionUtil.java     # 环状扭曲特效 (GraphicsLib)
│   │   ├── ARR_TemporalShellUtil.java  # 时流效果
│   │   ├── ARR_LocationUtil.java       # 舰船坐标系偏移
│   │   ├── ARR_StringTagUtil.java      # 字符串标签读取
│   │   ├── ControlSystem/
│   │   │   └── ShipControlSystem.java  # 精确舰船控制系统
│   │   ├── network/
│   │   │   └── SimpleSocketClient.java # Socket 通讯客户端
│   │   └── network_training/
│   │       └── ARR_StateSender.java    # 训练状态发送器
│   ├── weapons/
│   │   └── ScanBeamEffect.java         # 扫描激光传感器逻辑
│   └── world/
│       ├── ARR_NEXGenerate.java        # Nexerelin 兼容生成
│       └── systems/
│           └── Abyss.java              # 深井星系定义
└── impl/
    ├── combat/system/
    │   ├── DimensionDrivenField.java             # 维度驱动场（战斗版）
    │   └── DimensionDrivenField_TrainingVersion.java # 维度驱动场（训练版）
    └── hullmods/
        ├── ARR_ShowShipState.java     # 信息显示器
        ├── ARR_ShipController.java    # 远程控制器
        ├── ARR_CombatDataLink.java    # 战斗数据链
        ├── ARR_TrainingBridge.java    # 训练桥接器
        └── ARR_Duplication.java      # 多维并行器
```

## 构建

- **JDK**: 17 (`ms-17`)
- **源码**: `jars/src/`
- **产物**: `jars/Abyssal Redemption.jar`
- 使用 IntelliJ IDEA 打开 `.idea` 项目，编译后 Artifact 自动输出至 `jars/`

## 版权

北棱 (aikexue170) — GPL 协议，详见 [LICENSE](LICENSE)

## 开发笔记

开发过程中遇到的坑和解决方案，记录在 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。