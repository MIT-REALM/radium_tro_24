#!/bin/bash

conda activate architect_env

EXPNAME="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --no-repair --predict --savename intersection &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --L 10 --no-temper --no-repair --predict --savename intersection &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --no-repair --predict --savename intersection &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --no-repair --predict --savename intersection &

EXPNAME="radium" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --no-repair --predict --savename intersection &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --L 10 --no-temper --no-repair --predict --savename intersection &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --no-repair --predict --savename intersection &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --no-repair --predict --savename intersection &

wait;

EXPNAME="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --no-repair --predict --savename intersection &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --L 10 --no-temper --no-repair --predict --savename intersection &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --no-repair --predict --savename intersection &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --no-repair --predict --savename intersection &

EXPNAME="radium" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --no-repair --predict --savename intersection &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_gradients --L 10 --no-temper --no-repair --predict --savename intersection &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --no-repair --predict --savename intersection &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --no-temper --no-repair --predict --savename intersection &
