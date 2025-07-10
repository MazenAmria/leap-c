"""Main script to run experiments."""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

from leap_c.examples import create_env, create_controller
from leap_c.run import default_output_path, init_run
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.torch.rl.ppo_fop import PpoFopTrainer, PpoFopTrainerConfig


@dataclass
class RunPpoFopConfig:
    """Configuration for running PPO experiments."""

    env_name: str = "pointmass"
    controller_name: str = "pointmass"
    device: str = "cuda"  # or 'cpu'
    trainer: PpoFopTrainerConfig = field(default_factory=PpoFopTrainerConfig)


def run_ppo_fop(
    output_path: str | Path,
    env_name: str,
    controller_name: str,
    seed: int = 0,
    device: str = "cuda",
) -> float:
    cfg = PpoFopTrainerConfig()
    cfg.actor_mlp = MlpConfig(
        hidden_dims=(64, 64),
        activation="tanh",
        weight_init="orthogonal",
    )
    cfg.critic_mlp = MlpConfig(
        hidden_dims=(64, 64),
        activation="tanh",
        weight_init="orthogonal",
    )
    cfg.train_steps = 1_000_000
    cfg.val_interval = 10_000
    cfg.lr_q = 3e-4
    cfg.lr_pi = 3e-4
    cfg.anneal_lr = True
    cfg.update_epochs = 10
    cfg.num_steps = 2048
    cfg.num_mini_batches = 32
    cfg.clipping_epsilon = 0.2
    cfg.l_vf_weight = 0.25
    cfg.l_ent_weight = 0.001
    cfg.gamma = 0.99
    cfg.gae_lambda = 0.95
    cfg.clipping_epsilon = 0.2
    cfg.clip_value_loss = True
    cfg.normalize_advantages = True
    cfg.max_grad_norm = 0.5
    cfg.num_mini_batches = 32
    cfg.seed = seed

    trainer = PpoFopTrainer(
        val_env=create_env(env_name, render_mode="rgb_array"),
        train_envs=[create_env(env_name)],
        output_path=output_path,
        device=device,
        cfg=cfg,
        controller=create_controller(controller_name),
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="pointmass")
    parser.add_argument("--controller", type=str, default="pointmass")
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags=["ppo_fop", args.env, args.controller])

    run_ppo_fop(
        output_path,
        env_name=args.env,
        controller_name=args.controller,
        seed=args.seed,
        device=args.device,
    )
