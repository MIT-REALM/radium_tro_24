#!/bin/bash
for seed in 0 1 2 3
do
    EXPNAME="r1"  CUDA_VISIBLE_DEVICES=$seed, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --repair --predict --savename tro2-highway &
    EXPNAME="r0"  CUDA_VISIBLE_DEVICES=$seed, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --disable_gradients --no-temper --repair --predict --savename tro2-highway &
    EXPNAME="l2c" CUDA_VISIBLE_DEVICES=$seed, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --reinforce --no-temper --repair --predict --savename tro2-highway &
    EXPNAME="gda" CUDA_VISIBLE_DEVICES=$seed, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --disable_stochasticity --no-temper --repair --predict --savename tro2-highway &
    EXPNAME="gd"  CUDA_VISIBLE_DEVICES=$seed, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --disable_stochasticity --no-temper --repair --no-predict --savename tro2-highway &
done

# TODO run seed = 2