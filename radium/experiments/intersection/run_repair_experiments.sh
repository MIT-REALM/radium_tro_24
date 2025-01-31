#!/bin/bash
for seed in 0 1 2 3
do
    EXPNAME="radium" CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python radium/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed $seed --noise_scale 0.5 --num_rounds 15 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-6 --L 1.0 --grad_clip 1.0 --quench_rounds 5 --repair --predict --savename tro5-intersection &
    EXPNAME="rmh"    CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python radium/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed $seed --noise_scale 0.5 --num_rounds 15 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-6 --L 1.0 --grad_clip 1.0 --disable_gradients --no-temper --repair --predict --savename tro5-intersection &
    EXPNAME="l2c"    CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 python radium/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed $seed --noise_scale 0.5 --num_rounds 15 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-6 --L 1.0 --grad_clip 1.0 --reinforce --no-temper --repair --predict --savename tro5-intersection &
    EXPNAME="gda"    CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python radium/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed $seed --noise_scale 0.5 --num_rounds 15 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-6 --L 1.0 --grad_clip 1.0 --disable_stochasticity --no-temper --repair --predict --savename tro5-intersection &
    EXPNAME="gd"     CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 python radium/experiments/intersection/predict_and_mitigate.py --model_path results/neurips_submission/intersection/initial_policy.eqx --seed $seed --noise_scale 0.5 --num_rounds 15 --num_steps_per_round 10 --num_chains 5 --ep_mcmc_step_size 1e-2 --dp_mcmc_step_size 1e-6 --L 1.0 --grad_clip 1.0 --disable_stochasticity --no-temper --repair --no-predict --savename tro5-intersection &

    if [ $seed -eq 1 ]
    then
        wait;
    fi
done