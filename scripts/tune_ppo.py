import subprocess

import optuna
import torch

NUM_GPUS = torch.cuda.device_count()


def objective(trial):
    """Objective function for hyperparameter tuning."""

    gpu_id = trial.number % NUM_GPUS
    clipping_epsilon = trial.suggest_float("clipping_epsilon", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    update_epochs = trial.suggest_int("update_epochs", 5, 50, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.97)
    l_ent_weight = trial.suggest_float("l_ent_weight", 1e-4, 0.01, log=True)

    command = (
        f"python scripts/run_ppo.py "
        f"--device cuda:{gpu_id} "
        f"--steps 100000 "
        f"--clipping_epsilon {clipping_epsilon} "
        f"--lr {lr} "
        f"--update_epochs {update_epochs} "
        f"--gae_lambda {gae_lambda} "
        f"--l_ent_weight {l_ent_weight}"
        f" --wandb --wandb-team amriam-university-of-freiburg "
        f"--run_id {trial.number} "
    )

    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return float(result.stdout.decode().splitlines()[-1].strip())


study = optuna.create_study(
    direction="maximize", study_name="ppo_hyperparameter_tuning"
)
study.optimize(objective, n_trials=200, n_jobs=NUM_GPUS, show_progress_bar=True)

study.trials_dataframe().to_csv("ppo_hyperparameter_tuning_results.csv", index=False)

