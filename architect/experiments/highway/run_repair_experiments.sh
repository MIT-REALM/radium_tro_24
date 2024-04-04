#!/bin/bash
for seed in 0 1 2 3
do
    gpu=$seed
    
    EXPNAME="r1"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 1.0 --num_rounds 10 --repair --predict --quench_rounds 0 --savename tro5-highway &
    EXPNAME="r0"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 1.0 --num_rounds 10 --disable_gradients --no-temper --repair --predict --savename tro5-highway &
    EXPNAME="l2c" CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 1.0 --num_rounds 10 --reinforce --no-temper --repair --predict --savename tro5-highway &
    EXPNAME="gda" CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 1.0 --num_rounds 10 --disable_stochasticity --no-temper --repair --predict --savename tro5-highway &
    EXPNAME="gd"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 1.0 --num_rounds 10 --disable_stochasticity --no-temper --repair --no-predict --savename tro5-highway &
done