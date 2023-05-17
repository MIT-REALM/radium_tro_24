#!/bin/bash

EXPNAME="radium" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_gradients --L 10 --no-temper --repair --predict --savename highway &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --reinforce --no-temper --repair --predict --savename highway &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_stochasticity --no-temper --repair --predict --savename highway &

EXPNAME="radium" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_gradients --L 10 --no-temper --repair --predict --savename highway &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --reinforce --no-temper --repair --predict --savename highway &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_stochasticity --no-temper --repair --predict --savename highway &

EXPNAME="radium" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_gradients --L 10 --no-temper --repair --predict --savename highway &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --reinforce --no-temper --repair --predict --savename highway &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_stochasticity --no-temper --repair --predict --savename highway &

EXPNAME="radium" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_gradients --L 10 --no-temper --repair --predict --savename highway &
EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --reinforce --no-temper --repair --predict --savename highway &
EXPNAME="gd"     CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --disable_stochasticity --no-temper --repair --predict --savename highway &


EXPNAME="radium" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 0 --num_rounds 5 --quench_rounds 3 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="radium" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 1 --num_rounds 5 --quench_rounds 3 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="radium" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 2 --num_rounds 5 --quench_rounds 3 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &
EXPNAME="radium" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/old/highway_lqr/initial_policy.eqx --seed 3 --num_rounds 5 --quench_rounds 3 --num_steps_per_round 5 --num_chains 5  --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-2 --L 10 --repair --predict --savename highway &