"""Main script to run experiments."""
from dataclasses import dataclass, field
from pathlib import Path

from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.env import CartPoleEnv
from leap_c.run import init_run, create_parser, default_output_path
from leap_c.torch.rl.ppo_fop import PpoFopTrainer, PpoFopTrainerConfig


@dataclass
class RunPpoConfig:
    """Configuration for running PPO experiments."""

    device: str = "cuda"  # or 'cpu'
    trainer: PpoFopTrainerConfig = field(default_factory=PpoFopTrainerConfig)


def run_ppo(
    output_path: str | Path, seed: int = 0, device: str = "cuda"
) -> float:
    cfg = RunPpoConfig(device=device)
    cfg.trainer.seed = seed

    trainer = PpoFopTrainer(
        val_env=CartPoleEnv(render_mode="rgb_array"),
        train_env=[CartPoleEnv()],
        output_path=output_path,
        device=args.device,
        cfg=cfg.trainer,
        controller=CartPoleController()
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags={"trainer": "ppo"})

    run_ppo(output_path, seed=args.seed, device=args.device)
