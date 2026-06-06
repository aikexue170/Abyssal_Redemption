================================================================================
  PPO/DQN 强化学习框架 — 废弃归档
  归档日期: 2026-06-06
  归档原因: 重构强化学习框架，旧代码架构问题多，全部重做
================================================================================

【目录结构】
old_ppo_framework/
├── README_ARCHIVE.txt          # 本说明文件
├── java_source/                # Java 源代码 (已从 jars/src/ 移除)
│   ├── data/scripts/utils/PPO/
│   │   └── PPOClient_30state.java         # PPO的TCP客户端 (单例, port 65432)
│   └── impl/hullmods/
│       ├── PPOTrainer_30state.java        # 离散动作训练hullmod (8动作)
│       ├── PPOTrainer_Continuous.java      # 连续动作训练hullmod (move/turn/strafe)
│       └── PPOUser_30state.java           # 离散动作推理hullmod (8动作)
├── compiled_classes/           # 编译产物 (已从 out/production/ 移除)
│   ├── data/scripts/utils/PPO/
│   │   ├── PPOClient_30state.class
│   │   └── PPOClient_30state_Holder.class  (内部类)
│   └── impl/hullmods/
│       ├── PPOTrainer_30state.class
│       ├── PPOTrainer_Continuous.class
│       └── PPOUser_30state.class
└── python_trainer/             # Python端训练代码 (原 归档代码/PPO_python_trainer/)
    ├── PPO_framework.py                    # PPO算法核心框架
    ├── PPO_train_in_starsector.py          # 离散动作训练脚本
    ├── PPO_train_in_starsector_continuous.py # 连续动作训练脚本
    ├── PPO_train_in_virtual_env.py         # 虚拟环境训练脚本
    ├── PPO_use_in_starsector.py            # 推理/使用脚本
    ├── PPO_virtual_env.py                  # 虚拟对战环境
    ├── ppo_continuous_model_final.pth      # 连续模型权重(已训练)
    ├── replay_buffer.py                    # 经验回放缓冲
    ├── ship_controller_test.py             # 控制器测试
    ├── models/                             # 训练过程中的模型保存点
    ├── previous_models/                    # 更早的模型版本
    └── logs/                               # 训练日志

================================================================================

【关联数据文件变更】
以下 CSV 条目已从 data/hullmods/hull_mods.csv 中移除:

  原来第2行: DQN训练器, DQN_trainer → impl.hullmods.PPOTrainer_Continuous
  原来第3行: DQN控制器, DQN_user     → impl.hullmods.PPOUser_30state

【保留未归档的关键文件】
  jars/src/data/scripts/utils/PPO/SimpleSocketClient.java  — 通用TCP Socket客户端
    (被 ARR_ShipController.java 依赖，继续保留)

================================================================================

【架构问题总结 (为何废弃)】
1. PPOClient_30state 混用了CSV端的Socket通信方式 (简单send/readLine)
   与 SimpleSocketClient 的队列+线程方式不一致，架构混乱
2. PPOTrainer_30state 和 PPOUser_30state 代码几乎完全相同 (90%重复)
   差异仅在于是否应用 TemporalShell，应合并
3. PPOTrainer_Continuous 内部自建了 SimpleSocketClient + ShipControlSystem
   与 PPOClient_30state/POTrainer_30state 使用不同的通信和运动控制方式
4. PPOClient_30state 同时负责通信+状态显示+训练统计，职责耦合太重
5. CSV中错误标注为 "DQN" (应是PPO)，命名混淆
6. 通信间隔不一致 (0.1s vs 0.01s)，状态维度不统一
7. 没有抽象出通用的 RLClient 接口和 Trainer/User 基础类

================================================================================
