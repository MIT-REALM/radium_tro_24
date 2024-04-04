# Run small experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 5 --num_chains 5 --grad_clip 100 --failure_level 10.0 --L 5 --ep_mcmc_step_size 1e-5 --dp_mcmc_step_size 1e-5 --max_wind_thrust 1.0 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 5 --num_chains 5 --grad_clip 100 --failure_level 10.0 --L 5 --ep_mcmc_step_size 1e-5 --dp_mcmc_step_size 1e-5 --max_wind_thrust 1.0 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 5 --num_chains 5 --grad_clip 100 --failure_level 10.0 --L 5 --ep_mcmc_step_size 1e-5 --dp_mcmc_step_size 1e-5 --max_wind_thrust 1.0 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 5 --num_chains 5 --grad_clip 100 --failure_level 10.0 --L 5 --ep_mcmc_step_size 1e-5 --dp_mcmc_step_size 1e-5 --max_wind_thrust 1.0 --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 5 --num_chains 5 --grad_clip 100 --failure_level 10.0 --L 5 --ep_mcmc_step_size 1e-5 --dp_mcmc_step_size 1e-5 --max_wind_thrust 1.0 --temper --seed $seed &
done

wait;

# Run large experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 10 --L 5 --num_chains 5 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --failure_level 10.0 --grad_clip 100 --max_wind_thrust 1.0 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 10 --L 5 --num_chains 5 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --failure_level 10.0 --grad_clip 100 --max_wind_thrust 1.0 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 10 --L 5 --num_chains 5 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --failure_level 10.0 --grad_clip 100 --max_wind_thrust 1.0 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 10 --L 5 --num_chains 5 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --failure_level 10.0 --grad_clip 100 --max_wind_thrust 1.0 --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/formation2d/solve.py --n 10 --L 5 --num_chains 5 --ep_mcmc_step_size 1e-4 --dp_mcmc_step_size 1e-4 --failure_level 10.0 --grad_clip 100 --max_wind_thrust 1.0 --temper --seed $seed &

    # wait;
done
