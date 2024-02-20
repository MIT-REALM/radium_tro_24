#!/bin/bash
for seed in 0 1 2 3
do
    gpu=$seed
    
    EXPNAME="r1"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 25 --repair --predict --quench_rounds 10 --savename tro3-highway &
    EXPNAME="r0"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 25 --disable_gradients --no-temper --repair --predict --savename tro3-highway &
    EXPNAME="l2c" CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 25 --reinforce --no-temper --repair --predict --savename tro3-highway &
    EXPNAME="gda" CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 25 --disable_stochasticity --no-temper --repair --predict --savename tro3-highway &
    EXPNAME="gd"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 25 --disable_stochasticity --no-temper --repair --no-predict --savename tro3-highway &
done