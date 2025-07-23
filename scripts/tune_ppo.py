from argparse import ArgumentParser
from pathlib import Path

import neps

from leap_c.examples import create_env
from leap_c.run import init_run
from leap_c.torch.nn.mlp import MlpConfig
from leap_c.torch.rl.ppo import PpoTrainer, PpoTrainerConfig


def objective(
    pipeline_directory: Path,
    clipping_epsilon: float,
    lr: float,
    update_epochs: int,
    gae_lambda: float,
    l_ent_weight: float,
    steps: int,
    env: str = "pointmass",
    device: str = "cpu",
    wandb: bool = False,
    wandb_team: dict | None = None,
) -> float:
    """Objective function for hyperparameter tuning."""

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
    cfg.train_steps = steps
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
    cfg.seed = 0

    if wandb and wandb_team is not None:
        cfg.log.wandb_logger = True
        cfg.log.wandb_init_kwargs = {
            "project": "leap-c",
            "entity": wandb_team,
            "name": pipeline_directory.parts[-1],
            "config": cfg
        }

    trainer = PpoTrainer(
        val_env=create_env(env, difficulty="easy", render_mode="rgb_array"),
        train_envs=[create_env(env, difficulty="easy")],
        output_path=pipeline_directory,
        device=device,
        cfg=cfg,
    )
    init_run(trainer, cfg, pipeline_directory)

    return -trainer.run()


args = ArgumentParser()
args.add_argument("--env", type=str, default="pointmass")
args.add_argument("--device", type=str, default="cpu")
args.add_argument("--wandb", action="store_true", default=False)
args.add_argument("--wandb-team", type=str, default=None)
args.add_argument("--output-path", type=str, default="tune_ppo_output")
args = args.parse_args()

pipeline_space = {
    "clipping_epsilon": neps.Float(0.1, 0.3),
    "lr": neps.Float(1e-5, 5e-4, log=True),
    "update_epochs": neps.Integer(5, 50, log=True),
    "gae_lambda": neps.Float(0.9, 0.97),
    "l_ent_weight": neps.Float(1e-4, 0.01, log=True),
    "steps": neps.Integer(50_000, 1_000_000, is_fidelity=True),
}


def evaluate_pipeline(pipeline_directory: Path, **kwargs):
    return objective(
        pipeline_directory,
        env=args.env,
        device=args.device,
        wandb=args.wandb,
        wandb_team=args.wandb_team,
        **kwargs,
    )


neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    max_evaluations_total=160,
    root_directory=args.output_path,
    optimizer=neps.algorithms.async_hb,
    continue_until_max_evaluation_completed=True, 
    overwrite_working_directory=False,
)
