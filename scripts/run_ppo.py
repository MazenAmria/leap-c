"""Main script to run experiments."""

import os
from pathlib import Path

import optuna
import torch

from leap_c.examples import create_env
from leap_c.run import create_parser, default_output_path, init_run
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.torch.rl.ppo import PpoTrainer, PpoTrainerConfig


def run_ppo(
    clipping_epsilon: float,
    lr: float,
    update_epochs: int,
    gae_lambda: float,
    l_ent_weight: float,
    output_path: str | Path,
    seed: int = 0,
    env: str = "pointmass",
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
    cfg.lr_q = lr
    cfg.lr_pi = lr
    cfg.anneal_lr = True
    cfg.update_epochs = update_epochs
    cfg.num_steps = 2048
    cfg.num_mini_batches = 32
    cfg.clipping_epsilon = clipping_epsilon
    cfg.l_vf_weight = 0.25
    cfg.l_ent_weight = l_ent_weight
    cfg.gamma = 0.99
    cfg.gae_lambda = gae_lambda
    cfg.clip_value_loss = True
    cfg.normalize_advantages = True
    cfg.max_grad_norm = 0.5
    cfg.num_mini_batches = 32
    cfg.seed = seed

    if wandb and wandb_kwargs is not None:
        cfg.log.wandb_logger = True
        cfg.log.wandb_init_kwargs = wandb_kwargs

    trainer = PpoTrainer(
        val_env=create_env(env, difficulty="easy", render_mode="rgb_array"),
        train_envs=[create_env(env, difficulty="easy")],
        output_path=output_path,
        device=device,
        cfg=cfg,
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--env", type=str, default="pointmass")
    args = parser.parse_args()

    NUM_GPUS = torch.cuda.device_count()

    def objective(trial) -> float:
        gpu_id = trial.number % NUM_GPUS
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        output_path = default_output_path(seed=args.seed, tags=["ppo", args.env])

        wandb_kwargs = None
        if args.wandb:
            wandb_kwargs = {
                "project": "leap-c",
                "entity": args.wandb_team,
                "name": f"ppo_env_{args.env}_seed_{args.seed}_run_{trial.number}",
                "reinit": True,
            }

        clipping_epsilon = trial.suggest_float("clipping_epsilon", 0.1, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        update_epochs = trial.suggest_int("update_epochs", 5, 50, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.97)
        l_ent_weight = trial.suggest_float("l_ent_weight", 1e-4, 0.01, log=True)

        return run_ppo(
            clipping_epsilon=clipping_epsilon,
            lr=lr,
            update_epochs=update_epochs,
            gae_lambda=gae_lambda,
            l_ent_weight=l_ent_weight,
            output_path=output_path,
            seed=args.seed,
            env=args.env,
            device="cuda",
            wandb=args.wandb,
            wandb_kwargs=wandb_kwargs,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, n_jobs=NUM_GPUS)
