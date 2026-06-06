import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib

from PPO_virtual_env import SimpleShipEnv

matplotlib.use('Agg')  # 使用非交互式后端避免GUI问题
import matplotlib.pyplot as plt


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.state_dim = state_dim

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor - 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )

        # Critic - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        # 确保状态维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.shared(state)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy, value.squeeze()


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.entropies = []

        # 训练统计
        self.episode_rewards = []
        self.losses = []

    def get_action(self, state):
        """获取动作"""
        # 确保状态是正确维度的tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)

        # 检查状态维度
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        return self.policy.get_action(state)

    def store_transition(self, state, action, log_prob, value, reward, done, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.entropies.append(entropy)

    def compute_advantages(self, next_value):
        if not self.rewards:
            return [], []

        advantages = []
        returns = []

        # 将数据转移到CPU进行numpy计算
        rewards = np.array(self.rewards)
        values = np.array([v.item() for v in self.values] + [next_value.item()])
        dones = np.array(self.dones)

        # 计算GAE (Generalized Advantage Estimation)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = advantages + values[:-1]

        # 归一化优势
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def update(self, next_value):
        if len(self.states) < self.batch_size:
            return 0, 0, 0

        advantages, returns = self.compute_advantages(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 计算新策略的动作概率
                logits, values = self.policy(batch_states)
                new_probs = torch.softmax(logits, dim=-1)
                new_dist = torch.distributions.Categorical(new_probs)
                new_log_probs = new_dist.log_prob(batch_actions)
                entropy = new_dist.entropy().mean()

                # 概率比
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # 裁剪的PPO目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic损失 (MSE)
                critic_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()

                # 总损失
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # 更新
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy.item())

        # 清空缓冲区
        self.clear_buffer()

        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_entropy = np.mean(entropy_losses) if entropy_losses else 0

        return avg_actor_loss, avg_critic_loss, avg_entropy

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.entropies = []

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_ppo():
    # 创建环境
    env = SimpleShipEnv()

    # 动态获取状态维度
    test_state = env.reset()
    state_dim = len(test_state)
    action_dim = 7  # 7个离散动作

    print(f"检测到状态维度: {state_dim}")

    # 创建PPO智能体
    agent = PPOAgent(state_dim, action_dim)

    # 训练参数
    episodes = 5000
    max_steps = 1000
    save_interval = 100
    render_interval = 100

    # 训练统计
    episode_rewards = []
    success_rates = []
    losses = []

    print("开始PPO训练...")
    print(f"每 {render_interval} 回合会显示一次训练过程")

    try:
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            success = False

            # 决定是否渲染这个回合
            render_this_episode = (episode % render_interval == 0)

            if render_this_episode:
                print(f"\n=== 回合 {episode} 演示 (按ESC跳过) ===")

            for step in range(max_steps):
                # 选择动作
                action, log_prob, entropy, value = agent.get_action(state)

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                agent.store_transition(state, action, log_prob, value, reward, done, entropy)

                state = next_state
                episode_reward += reward

                if env.success:
                    success = True

                # 如果需要渲染，显示当前状态
                if render_this_episode:
                    env.render()

                    # 添加状态信息显示
                    font = pygame.font.Font(None, 24)
                    info_text = [
                        f"回合: {episode}",
                        f"步骤: {step}",
                        f"当前奖励: {reward:.3f}",
                        f"累计奖励: {episode_reward:.1f}",
                        f"动作: {['W', 'S', 'A', 'D', 'Q', 'E', '无'][action]}",
                        f"成功: {'是' if success else '否'}"
                    ]

                    for i, text in enumerate(info_text):
                        text_surface = font.render(text, True, (255, 255, 255))
                        env.screen.blit(text_surface, (10, 10 + i * 25))

                    pygame.display.flip()

                    # 检查是否按ESC跳过演示
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                render_this_episode = False
                                print("跳过演示...")

                    # 控制渲染速度，不要太快
                    pygame.time.delay(50)  # 20 FPS

                if done:
                    break

            if render_this_episode:
                print(f"演示结束 - 奖励: {episode_reward:.1f}, 成功: {success}")

            # 回合结束，计算最终状态的价值
            with torch.no_grad():
                _, next_value = agent.policy(torch.FloatTensor(state).to(agent.device))

            # 更新策略
            actor_loss, critic_loss, entropy_loss = agent.update(next_value)

            # 记录统计
            episode_rewards.append(episode_reward)
            success_rates.append(1 if success else 0)
            if actor_loss > 0:
                losses.append((actor_loss, critic_loss, entropy_loss))

            # 每10轮输出一次统计信息
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
                success_rate = np.mean(success_rates[-10:]) * 100 if len(success_rates) >= 10 else (
                    100 if success else 0)

                print(f"回合 {episode}/{episodes} | "
                      f"奖励: {episode_reward:.1f} (平均: {avg_reward:.1f}) | "
                      f"成功率: {success_rate:.1f}% | "
                      f"步骤: {step + 1}")

                if losses:
                    avg_actor_loss = np.mean([l[0] for l in losses[-10:]])
                    avg_critic_loss = np.mean([l[1] for l in losses[-10:]])
                    avg_entropy = np.mean([l[2] for l in losses[-10:]])
                    print(
                        f"    损失 - Actor: {avg_actor_loss:.4f}, Critic: {avg_critic_loss:.4f}, Entropy: {avg_entropy:.4f}")

            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                agent.save_model(f"ppo_ship_model_episode_{episode}.pth")
                print(f"模型已保存: ppo_ship_model_episode_{episode}.pth")

    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        agent.save_model("ppo_ship_model_final.pth")
        print("训练完成，最终模型已保存")

        # 绘制训练曲线
        plot_training(episode_rewards, success_rates, losses)

def test_episode(env, agent, render=True, max_steps=1000):
    """测试一个回合"""
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        with torch.no_grad():
            action, _, _, _ = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()
            pygame.time.delay(20)

        state = next_state

        if done or env.success:
            break

    return total_reward, step + 1, env.success


def plot_training(episode_rewards, success_rates, losses):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))

    # 奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.6)
    # 添加移动平均
    if len(episode_rewards) > 10:
        window = min(50, len(episode_rewards) // 10)
        rewards_smooth = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), rewards_smooth, 'r-', linewidth=2,
                 label=f'{window}ep moving avg')
        plt.legend()
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 成功率曲线
    plt.subplot(1, 3, 2)
    success_rates_percent = [s * 100 for s in success_rates]
    plt.plot(success_rates_percent, alpha=0.6)
    if len(success_rates) > 10:
        window = min(50, len(success_rates) // 10)
        success_smooth = np.convolve(success_rates_percent, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(success_rates)), success_smooth, 'r-', linewidth=2,
                 label=f'{window}ep moving avg')
        plt.legend()
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)

    # 损失曲线
    if losses:
        plt.subplot(1, 3, 3)
        actor_losses = [l[0] for l in losses]
        critic_losses = [l[1] for l in losses]

        plt.plot(actor_losses, alpha=0.6, label='Actor Loss')
        plt.plot(critic_losses, alpha=0.6, label='Critic Loss')

        # 平滑损失曲线
        if len(actor_losses) > 10:
            window = min(50, len(actor_losses) // 10)
            actor_smooth = np.convolve(actor_losses, np.ones(window) / window, mode='valid')
            critic_smooth = np.convolve(critic_losses, np.ones(window) / window, mode='valid')

            plt.plot(range(window - 1, len(actor_losses)), actor_smooth, 'b-', linewidth=2, label='Actor (smooth)')
            plt.plot(range(window - 1, len(critic_losses)), critic_smooth, 'orange', linewidth=2,
                     label='Critic (smooth)')

        plt.legend()
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存为 'training_curves.png'")
    plt.close()

if __name__ == "__main__":
    # 可以选择训练或测试模式
    mode = "train"  # 改为 "test" 来测试现有模型

    if mode == "train":
        train_ppo()
    else:
        # 测试模式
        env = SimpleShipEnv()
        state_dim = 10
        action_dim = 7

        agent = PPOAgent(state_dim, action_dim)
        agent.load_model("ppo_ship_model_final.pth")  # 加载训练好的模型

        print("开始测试...")
        for i in range(5):
            test_episode(env, agent, render=True, max_steps=1000)

        env.close()