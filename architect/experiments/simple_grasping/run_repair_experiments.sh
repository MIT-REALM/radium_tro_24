# Mug

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type mug --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type mug --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type mug --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type mug --savename grasping &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type mug --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type mug --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type mug --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type mug --savename grasping &

wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type mug --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type mug --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type mug --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type mug --savename grasping &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type mug --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type mug --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type mug --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type mug --savename grasping &

# box
wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type box --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type box --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type box --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type box --savename grasping &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type box --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type box --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type box --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type box --savename grasping &

wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type box --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type box --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type box --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type box --savename grasping &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type box --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type box --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type box --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type box --savename grasping &

# bowl
wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type bowl --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type bowl --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 0 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type bowl --savename grasping &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type bowl --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type bowl --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 1 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type bowl --savename grasping &

wait;

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type bowl --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type bowl --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 2 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type bowl --savename grasping &

EXP="radium"     CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --object_type bowl --savename grasping &
EXP="rocus"      CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --L 10 --repair --predict --disable_gradients --no-temper --object_type bowl --savename grasping &
EXP="l2c"        CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --reinforce --no-temper --repair --predict --object_type bowl --savename grasping &
EXP="gd"         CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 python architect/experiments/simple_grasping/predict_and_mitigate.py --model_path results/grasping_mug/initial_policy.eqx --seed 3 --num_rounds 5 --num_steps_per_round 5 --num_chains 5 --ep_mcmc_step_size 1e-3 --dp_mcmc_step_size 1e-3 --disable_stochasticity --repair --predict --no-temper --object_type bowl --savename grasping &

