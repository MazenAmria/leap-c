from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, NamedTuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Normal

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.extractor import Extractor, IdentityExtractor
from leap_c.torch.nn.gaussian import BoundedTransform
from leap_c.torch.nn.mlp import MLP, MlpConfig
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.ppo import PpoCritic, PpoTrainerConfig, ClippedSurrogateLoss, ValueSquaredErrorLoss
from leap_c.trainer import Trainer
from leap_c.utils.gym import seed_env, wrap_env


@dataclass(kw_only=True)
class PpoFopTrainerConfig(PpoTrainerConfig):
    entropy_correction: bool = False


class PpoFopActorOutput(NamedTuple):
    param: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor
    status: torch.Tensor
    ctx: Any | None

    def select(self, mask: torch.Tensor) -> "PpoFopActorOutput":
        return PpoFopActorOutput(
            self.param[mask],
            self.log_prob[mask],
            self.entropy[mask],
            None,  # type:ignore
            self.action[mask],
            self.status[mask],
            None,
        )


class PpoFopActor(nn.Module):
    def __init__(
            self,
            extractor_cls: Type[Extractor],
            observation_space: spaces.Space,
            mlp_cfg: MlpConfig,
            controller: ParameterizedController,
            correction: bool = True,
    ):
        super().__init__()

        self.extractor = extractor_cls(observation_space)

        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=controller.param_space.shape[0],  # type: ignore
            mlp_cfg=mlp_cfg,
        )

        self.log_std = nn.Parameter(torch.zeros(1, controller.param_space.shape[0]))

        self.controller = controller
        self.correction = correction

    def forward(self, obs: torch.Tensor, deterministic: bool = False, param=None, ctx=None):
        e = self.extractor(obs)
        mean = self.mlp(e)
        std = self.log_std.expand_as(mean).exp()

        probs = Normal(mean, std)

        if param is None:
            param = probs.mode if deterministic else probs.sample()

        log_prob = probs.log_prob(param).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)

        ctx, action = self.controller(obs, param, ctx=ctx)

        j = self.controller.jacobian_action_param(ctx)
        if j is not None and self.correction:
            jtj = j @ j.transpose(1, 2)
            correction = (
                torch.det(jtj + 1e-3 * torch.eye(jtj.shape[1], device=jtj.device))
                .sqrt()
                .log()
            )
            log_prob -= correction.unsqueeze(1)
            entropy += correction.unsqueeze(1)

        return PpoFopActorOutput(
            param=param,
            log_prob=log_prob,
            entropy=entropy,
            stats=ctx.log,
            action=action,
            status=ctx.status,
            ctx=ctx,
        )


class PpoFopTrainer(Trainer):
    def __init__(
            self,
            cfg: PpoFopTrainerConfig,
            val_env: gym.Env,
            output_path: str | Path,
            device: str,
            train_envs: List[gym.Env],
            controller: ParameterizedController,
            extractor_cls: Type[Extractor] = IdentityExtractor,
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            cfg: The configuration for the trainer.
        """
        wrappers = [
            lambda env: gym.wrappers.FlattenObservation(env),
            lambda env: gym.wrappers.ClipAction(env), # TODO (Mazen): try SquashedTanh
            lambda env: gym.wrappers.NormalizeObservation(env), # TODO (Mazen): use ScalingExtractor instead
            lambda env: gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), None),
            lambda env: gym.wrappers.NormalizeReward(env, gamma=self.cfg.gamma),
            lambda env: gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)),
        ]
        super().__init__(cfg, val_env, output_path, device, wrappers)

        self.train_env = seed_env(gym.vector.SyncVectorEnv([
            lambda: wrap_env(train_envs[i])
            for i, env in enumerate(train_envs)
        ]), self.cfg.seed)

        self.q = PpoCritic(
            extractor_cls,
            self.train_env.single_observation_space,
            self.cfg.critic_mlp
        )
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr_q)
        self.q_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.q_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.cfg.train_steps // (self.cfg.num_steps * self.train_env.num_envs)
        )

        self.pi = PpoFopActor(
            extractor_cls,
            self.train_env.single_action_space,  # type: ignore
            self.train_env.single_observation_space,
            self.cfg.actor_mlp,
            controller=controller,
            correction=self.cfg.entropy_correction
        )
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=self.cfg.lr_pi)
        self.pi_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.pi_optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.cfg.train_steps // (self.cfg.num_steps * self.train_env.num_envs)
        )

        self.clipped_loss = ClippedSurrogateLoss(self.cfg.clipping_epsilon)
        self.value_loss = ValueSquaredErrorLoss(self.cfg.clip_value_loss, self.cfg.clipping_epsilon)

        self.buffer = ReplayBuffer(self.cfg.num_steps, device, collate_fn_map=controller.collate_fn_map)

    def train_loop(self) -> Iterator[int]:
        obs, _ = self.train_env.reset(seed=self.cfg.seed, options={"mode": "train"})

        while True:
            # region Rollout Collection
            policy_state = None
            obs_collate = torch.tensor(obs, device=self.device)

            with torch.no_grad():
                pi_output = self.pi(obs_collate, ctx=policy_state)
                value = self.q(obs_collate)

            value = value.cpu().numpy()
            action = pi_output.action.cpu().numpy()
            log_prob = pi_output.log_prob.cpu().numpy()
            entropy = pi_output.entropy.cpu().numpy()
            param = pi_output.param.cpu().numpy()

            self.report_stats("train_trajectory", {"action": action}, verbose=True, with_smoothing=False)

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            self.buffer.put((
                obs,
                param,
                log_prob,
                entropy,
                reward,
                obs_prime,
                is_terminated,
                np.logical_or(is_terminated, is_truncated),
                value,
                pi_output.ctx
            ))

            if "episode" in info:
                idx = info["_episode"].argmax()
                self.report_stats("train", {
                    "episodic_return": float(info["episode"]["r"][idx]),
                    "episodic_length": int(info["episode"]["l"][idx]),
                }, with_smoothing=False)
            # endregion

            obs = obs_prime
            policy_state = pi_output.ctx

            if (self.state.step + self.train_env.num_envs) % (self.cfg.num_steps * self.train_env.num_envs) == 0:
                # region Generalized Advantage Estimation (GAE)
                advantages = torch.zeros((self.cfg.num_steps, self.train_env.num_envs), device=self.device)
                returns = torch.zeros((self.cfg.num_steps, self.train_env.num_envs), device=self.device)
                with torch.no_grad():
                    for t in reversed(range(self.cfg.num_steps)):
                        _, _, _, _, reward, obs_prime, termination, done, value, _ = self.buffer[t]

                        reward = reward.squeeze(0)
                        obs_prime = obs_prime.squeeze(0)
                        termination = termination.squeeze(0)
                        done = done.squeeze(0)
                        value = value.squeeze(0)

                        value_prime = self.q(obs_prime)

                        # TD Error: δ = r + γ * V' - V
                        delta = reward + self.cfg.gamma * value_prime * (1.0 - termination) - value

                        # GAE: A = δ + γ * λ * A'
                        advantage_prime = advantages[t + 1] if t != self.cfg.num_steps - 1 \
                            else torch.zeros(self.train_env.num_envs, device=self.device)
                        advantages[t] = delta + self.cfg.gamma * self.cfg.gae_lambda \
                                        * (1.0 - done) * advantage_prime

                        # Returns: G = A + V
                        returns[t] = advantages[t] + value
                # endregion

                # region Loss Calculation and Parameter Optimization
                mini_batch_size = (self.cfg.num_steps * self.train_env.num_envs) // self.cfg.num_mini_batches
                indices = np.arange(self.cfg.num_steps * self.train_env.num_envs)
                for epoch in range(self.cfg.update_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, self.cfg.num_steps * self.train_env.num_envs, mini_batch_size):
                        end = start + mini_batch_size
                        mb_indices = indices[start:end]
                        observations, params, log_probs, entropy, _, _, _, _, values, ctx = self.buffer[mb_indices]

                        observations = observations.flatten(start_dim=0, end_dim=1)
                        params = params.flatten(start_dim=0, end_dim=1)
                        log_probs = log_probs.flatten(start_dim=0, end_dim=1)

                        mb_advantages = advantages[mb_indices].flatten()
                        mb_returns = returns[mb_indices].flatten()

                        if self.cfg.normalize_advantages:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        new_values = self.q(observations)
                        pi_output = self.pi(observations, param=params, ctx=ctx)

                        # Calculating Loss
                        l_clip = self.clipped_loss(pi_output.log_prob, log_probs, mb_advantages)
                        l_vf = self.value_loss(new_values, values, mb_returns)
                        l_ent = -pi_output.entropy.mean()

                        loss = l_clip + self.cfg.l_ent_weight * l_ent + self.cfg.l_vf_weight * l_vf

                        # Updating Parameters
                        for optim in self.optimizers:
                            optim.zero_grad()
                        loss.backward()
                        for optim in self.optimizers:
                            nn.utils.clip_grad_norm_(optim.param_groups[0]["params"], self.cfg.max_grad_norm)
                            optim.step()

                self.report_stats("train", {
                    "policy_loss": l_clip.item(),
                    "value_loss": l_vf.item(),
                    "entropy": entropy.mean().item(),
                    "learning_rate": self.q_optim.param_groups[0]["lr"]
                }, with_smoothing=False)
                # endregion

                if self.cfg.anneal_lr:
                    self.q_lr_scheduler.step()
                    self.pi_lr_scheduler.step()

                self.buffer.clear()

            yield self.train_env.num_envs

    def act(
            self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, None, dict[str, float]]:
        obs = self.buffer.collate([obs])
        with torch.no_grad():
            output = self.pi(obs, deterministic=deterministic, ctx=state)
        
        action = output.action.cpu().numpy()[0]
        return action, None, output.stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.q_optim, self.pi_optim]
