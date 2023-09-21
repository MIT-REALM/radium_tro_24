#!/bin/bash

conda activate architect_env


wait;


EXPNAME="mala_1" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --L 5 --repair --predict --savename iclr_intersection &
EXPNAME="mala_1" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --L 5 --repair --predict --savename iclr_intersection &
EXPNAME="mala_1" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --L 5 --repair --predict --savename iclr_intersection &
EXPNAME="mala_1" CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --L 5 --repair --predict --savename iclr_intersection &

EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --disable_gradients --L 5 --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --disable_gradients --L 5 --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --disable_gradients --L 5 --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="rmh" CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --disable_gradients --L 5 --no-temper --repair --predict --savename iclr_intersection &

EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="l2c" CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --reinforce --no-temper --repair --predict --savename iclr_intersection &

EXPNAME="gd"  CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 0 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --disable_stochasticity --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="gd"  CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 1 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --disable_stochasticity --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="gd"  CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 2 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --disable_stochasticity --no-temper --repair --predict --savename iclr_intersection &
EXPNAME="gd"  CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/intersection/predict_and_mitigate.py --model_path results/response_int/initial_policy.eqx --seed 3 --num_rounds 10 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-4 --grad_clip 1 --disable_stochasticity --no-temper --repair --predict --savename iclr_intersection &
