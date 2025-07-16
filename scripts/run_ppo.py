"""Main script to run experiments."""

from pathlib import Path

from leap_c.examples import create_env
from leap_c.run import create_parser, default_output_path, init_run
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.torch.rl.ppo import PpoTrainer, PpoTrainerConfig


def run_ppo(
    output_path: str | Path,
    seed: int = 0,
    env: str = "cartpole",
    device: str = "cuda",
    wandb: bool = False,
    wandb_kwargs: dict | None = None,
) -> float:
    cfg = PpoTrainerConfig()
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

    if wandb and wandb_kwargs is not None:
        cfg.log.wandb_logger = True
        cfg.log.wandb_init_kwargs = wandb_kwargs

    trainer = PpoTrainer(
        val_env=create_env(env, render_mode="rgb_array"),
        train_envs=[create_env(env)],
        output_path=output_path,
        device=device,
        cfg=cfg,
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--env", type=str, default="cartpole")
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags=["ppo", args.env])

    wandb_kwargs = None
    if args.wandb:
        wandb_kwargs = {
            "project": "leap-c",
            "entity": args.wandb_team,
            "name": f"ppo_env_{args.env}_seed_{args.seed}",
        }

    run_ppo(output_path, seed=args.seed, env=args.env, device=args.device, wandb=args.wandb, wandb_kwargs=wandb_kwargs)
