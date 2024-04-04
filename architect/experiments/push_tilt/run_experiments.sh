wait

CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 0 --disable_stochasticity --no-predict &  # GD no adv
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 0 --disable_stochasticity &  # GD
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 0 --disable_gradients --quench_rounds 0 &  # RMH
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 0 &  # MALA
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 0 --reinforce &  # REINFORCE

CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 1 --disable_stochasticity --no-predict &  # GD no adv
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 1 --disable_stochasticity &  # GD
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 1 --disable_gradients --quench_rounds 0 &  # RMH
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 1 &  # MALA
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 1 --reinforce &  # REINFORCE

CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 2 --disable_stochasticity --no-predict &  # GD no adv
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 2 --disable_stochasticity &  # GD
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 2 --disable_gradients --quench_rounds 0 &  # RMH
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 2 &  # MALA
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 2 --reinforce &  # REINFORCE

CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 3 --disable_stochasticity --no-predict &  # GD no adv
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 3 --disable_stochasticity &  # GD
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 3 --disable_gradients --quench_rounds 0 &  # RMH
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 3 &  # MALA
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python architect/experiments/push_tilt/solve.py --seed 3 --reinforce &  # REINFORCE

wait

CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/stress_test.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/mala &
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/stress_test.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/gd &
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/stress_test.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/no_predict/gd &
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/stress_test.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/rmh &
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/stress_test.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/reinforce &

# wait

## Training curves are now computed during training and logged to WandB
# CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/training_curves.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/mala &
# CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/training_curves.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/gd &
# CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/training_curves.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/no_predict/gd &
# CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/training_curves.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/rmh &
# CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python architect/experiments/push_tilt/training_curves.py --file_prefix results/f16/L_1.0e+01/50_samples/0_quench/temper10_chains/dp_1.0e-04/ep_1.0e-03/repair/predict/reinforce &

wait

# python architect/experiments/push_tilt/plot_training_curves.py

python architect/experiments/push_tilt/plot_stress_test_results.py 
