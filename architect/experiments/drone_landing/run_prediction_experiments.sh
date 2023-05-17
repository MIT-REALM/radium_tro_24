#!/bin/bash

conda activate architect_env

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --savename drone &
# EXP="radium_not" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --no-temper --savename drone &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --disable_gradients --no-temper --savename drone &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --reinforce --no-temper --no-repair --predict --savename drone &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 0 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --disable_stochasticity --no-repair --predict --no-temper --savename drone &

EXP="radium"     CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --savename drone &
# EXP="radium_not" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --no-temper --savename drone &
EXP="rocus"      CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --disable_gradients --no-temper --savename drone &
EXP="l2c"        CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --reinforce --no-temper --no-repair --predict --savename drone &
EXP="gd"         CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 1 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --disable_stochasticity --no-repair --predict --no-temper --savename drone &

# wait; 

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --savename drone &
# EXP="radium_not" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --no-temper --savename drone &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --disable_gradients --no-temper --savename drone &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --reinforce --no-temper --no-repair --predict --savename drone &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 2 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --disable_stochasticity --no-repair --predict --no-temper --savename drone &

EXP="radium"     CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --savename drone &
# EXP="radium_not" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --no-temper --savename drone &
EXP="rocus"      CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --L 10 --no-repair --predict --disable_gradients --no-temper --savename drone &
EXP="l2c"        CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --reinforce --no-temper --no-repair --predict --savename drone &
EXP="gd"         CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/drone_landing/predict_and_mitigate.py --model_path results/old/drone_landing_smooth/initial_policy.eqx --seed 3 --num_rounds 25 --num_steps_per_round 1 --num_chains 5 --T 70 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --num_trees 5 --disable_stochasticity --no-repair --predict --no-temper --savename drone &
