"""ppo.py —— 单文件全 GPU PPO (参考 CleanRL ppo.py 改写, 连续动作)。

设计要点:
- rollout buffer 全程为 CUDA tensor ([T, N, ...]), 不出 GPU, 无 numpy 往返。
- Actor: MLP -> 动作均值; log_std 为状态无关可学参数; Normal 采样后 clamp 到 [-1,1]
  (logprob 用未 clamp 的原样本计算, 与 CleanRL 一致)。
- Critic: 独立 MLP -> 标量价值。
- GAE(gae_lambda) + clipped surrogate + 价值裁剪 + 熵奖励 + 梯度裁剪。
- 截断 (超时) 回合用终态观测的价值自举; 真终止 (成功/出界) 价值为 0。
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return layers


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128,
                 init_log_std: float = -0.5, min_std: float = 0.1):
        super().__init__()
        self.min_std = min_std   # 标准差下限: 防探索崩溃 (历史教训: 熵崩后高 lr 打崩策略)
        # buffer 形式参与 clamp: torch.compile 会把 python 标量烘焙成图常量,
        # buffer 是图输入且地址固定, fill_ 就地改值对 eager/fusion/CUDA graphs 回放都生效,
        # 训练末段退火 min_std 时通过 set_min_std 动态调整。
        self.register_buffer("min_std_t", torch.tensor(float(min_std)))
        self.actor = nn.Sequential(*_mlp([obs_dim, hidden, hidden, act_dim]))
        self.critic = nn.Sequential(*_mlp([obs_dim, hidden, hidden, 1]))
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        # 正交初始化 (CleanRL 惯例)
        for m in list(self.actor) + list(self.critic):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)   # 输出层小增益
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = self.log_std.exp().clamp_min(self.min_std_t).clamp_max(2.0).expand_as(mean)
        return Normal(mean, std), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """返回 (clamp后动作, logp, value, 原样本raw)。
        raw 必须存进 rollout buffer: logp 是在 raw 上算的,
        update 时重算 logp 也要用 raw, 否则 clamp 造成 logp 失配。"""
        dist, value = self.forward(obs)
        raw = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(raw).sum(-1)
        return raw.clamp(-1.0, 1.0), logp, value, raw

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def set_min_std(self, v: float):
        """训练末段退火用: 同步 python 属性 (存档) 与 buffer (计算图输入)。"""
        self.min_std = float(v)
        self.min_std_t.fill_(float(v))


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, device: str = "cuda",
                 hidden: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_coef: float = 0.2, vf_coef: float = 0.5,
                 ent_coef: float = 0.0, max_grad_norm: float = 1.0,
                 update_epochs: int = 4, num_minibatches: int = 8,
                 init_log_std: float = -0.5, min_std: float = 0.1, seed: int = 0):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim, hidden, init_log_std, min_std).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_coef, self.vf_coef, self.ent_coef = clip_coef, vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs, self.num_minibatches = update_epochs, num_minibatches

    # ---------------- rollout 存储 ----------------

    def init_buffer(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int):
        d = self.device
        self.buf = dict(
            obs=torch.zeros(num_steps, num_envs, obs_dim, device=d),
            act=torch.zeros(num_steps, num_envs, act_dim, device=d),
            logp=torch.zeros(num_steps, num_envs, device=d),
            rew=torch.zeros(num_steps, num_envs, device=d),
            done=torch.zeros(num_steps, num_envs, device=d),      # terminated | truncated
            trunc=torch.zeros(num_steps, num_envs, device=d),     # 仅超时截断
            val=torch.zeros(num_steps, num_envs, device=d),
            tval=torch.zeros(num_steps, num_envs, device=d),      # 截断回合的终态价值
        )
        self.num_steps, self.num_envs = num_steps, num_envs

    # ---------------- GAE + 更新 ----------------

    def compute_returns(self, next_value: torch.Tensor):
        """next_value: 当前 (post-reset) 观测的价值 [N]。返回 adv, ret。"""
        b, T = self.buf, self.num_steps
        adv = torch.zeros_like(b["rew"])
        lastgae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(T)):
            if t == T - 1:
                nextval = next_value
            else:
                nextval = b["val"][t + 1]
            # 截断: 用终态价值自举; 真终止: 不传播
            nextval = torch.where(b["trunc"][t].bool(), b["tval"][t], nextval)
            nonterminal = 1.0 - b["done"][t]
            delta = b["rew"][t] + self.gamma * nextval * nonterminal - b["val"][t]
            lastgae = delta + self.gamma * self.gae_lambda * nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + b["val"]
        return adv, ret

    def update(self, adv: torch.Tensor, ret: torch.Tensor) -> dict:
        b = self.buf
        T, N = self.num_steps, self.num_envs
        flat_obs = b["obs"].reshape(T * N, -1)
        flat_act = b["act"].reshape(T * N, -1)
        flat_logp = b["logp"].reshape(T * N)
        flat_adv = adv.reshape(T * N)
        flat_ret = ret.reshape(T * N)
        flat_val = b["val"].reshape(T * N)

        idx = torch.arange(T * N, device=self.device)
        stats = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, clipfrac=0.0, n=0)
        for _ in range(self.update_epochs):
            perm = idx[torch.randperm(T * N, device=self.device)]
            for mb in perm.chunk(self.num_minibatches):
                dist, value = self.net(flat_obs[mb])
                logp = dist.log_prob(flat_act[mb]).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                logratio = logp - flat_logp[mb]
                ratio = logratio.exp()

                madv = flat_adv[mb]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)

                pg1 = -madv * ratio
                pg2 = -madv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_clipped = flat_val[mb] + (value - flat_val[mb]).clamp(
                    -self.clip_coef, self.clip_coef)
                vf_loss = 0.5 * torch.max((value - flat_ret[mb]) ** 2,
                                          (v_clipped - flat_ret[mb]) ** 2).mean()

                loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                stats["pg"] += pg_loss.item(); stats["vf"] += vf_loss.item()
                stats["ent"] += ent.item(); stats["kl"] += kl.item()
                stats["clipfrac"] += clipfrac.item(); stats["n"] += 1

        return {k: (v / stats["n"] if k != "n" else v) for k, v in stats.items()}

    # ---------------- 存档 ----------------

    def save(self, path: str, extra: dict | None = None):
        payload = dict(state_dict=self.net.state_dict(),
                       obs_dim=self.net.actor[0].in_features,
                       act_dim=self.net.log_std.numel(),
                       min_std=self.net.min_std)
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PPO":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], device=device)
        # strict=False: 旧存档无 min_std_t buffer (v7 引入), 缺失时保留构造默认值
        agent.net.load_state_dict(ckpt["state_dict"], strict=False)
        return agent
