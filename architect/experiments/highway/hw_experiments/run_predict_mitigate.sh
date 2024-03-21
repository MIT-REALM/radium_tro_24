# Predict adversarial examples for the base policy
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python architect/experiments/highway/hw_experiments/predict_and_mitigate.py \
    --mlp_path architect/experiments/highway/hw_experiments/base/mlp.eqx \
    --ego_traj_path architect/experiments/highway/hw_experiments/base/ego_traj.eqx \
    --seed 0 --L 1.0 --failure_level 2.2 --num_rounds 10 --num_steps_per_round 10 \
    --num_chains 10 \
    --predict --quench_rounds 0 --savename tro_hardware_hw

# Predict repair the policy
CUDA_VISIBLE_DEVICES=1, XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python architect/experiments/highway/hw_experiments/predict_and_mitigate.py \
    --mlp_path architect/experiments/highway/hw_experiments/base/mlp.eqx \
    --ego_traj_path architect/experiments/highway/hw_experiments/base/ego_traj.eqx \
    --seed 0 --L 1.0 --failure_level 2.2 --num_rounds 10 --num_steps_per_round 10 \
    --num_chains 10 \
    --predict --repair --quench_rounds 2 --savename tro_hardware_hw