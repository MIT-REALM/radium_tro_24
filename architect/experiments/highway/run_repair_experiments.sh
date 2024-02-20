#!/bin/bash
for seed in 0 1 2 3
do
    if [ $seed -eq 0 ] || [ $seed -eq 2 ]; then
        gpu=1
    else
        gpu=3
    fi
    
    EXPNAME="r1"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 20 --repair --predict --quench_rounds 5 --savename tro2-highway &
    EXPNAME="r0"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 20 --disable_gradients --no-temper --repair --predict --savename tro2-highway &
    EXPNAME="l2c" CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 20 --reinforce --no-temper --repair --predict --savename tro2-highway &
    EXPNAME="gda" CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 20 --disable_stochasticity --no-temper --repair --predict --savename tro2-highway &
    EXPNAME="gd"  CUDA_VISIBLE_DEVICES=$gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/highway/predict_and_mitigate.py --model_path results/highway/initial_policy.eqx --seed $seed --L 10.0 --num_rounds 20 --disable_stochasticity --no-temper --repair --no-predict --savename tro2-highway &    

    if [ $seed -eq 1 ]; then
        wait;
    fi
done

# TODO run seed = 2