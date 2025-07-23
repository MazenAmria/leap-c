#!/bin/bash

python scripts/tune_ppo.py --env="pointmass" --device="cuda:0" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:1" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:2" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:3" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:4" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:5" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:6" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
python scripts/tune_ppo.py --env="pointmass" --device="cuda:7" --wandb --wandb-team="amriam-university-of-freiburg" --output-path="tuning/ppo/pointmass" &
