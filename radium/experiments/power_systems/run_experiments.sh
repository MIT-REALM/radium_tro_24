# Run small experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --failure_level 4.0 --num_rounds 50 --L 500 --grad_clip 100 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --failure_level 4.0 --num_rounds 50 --L 500 --grad_clip 100 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --failure_level 4.0 --num_rounds 50 --L 500 --grad_clip 100 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --failure_level 4.0 --num_rounds 50 --L 500 --grad_clip 100 --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --failure_level 4.0 --num_rounds 50 --L 500 --grad_clip 100 --seed $seed --quench_rounds 25 &
done

wait;

# Run large experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --case_name case57 --failure_level 6.0 --num_rounds 50 --L 500 --grad_clip 100 --ep_mcmc_step_size 1e-4 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --case_name case57 --failure_level 6.0 --num_rounds 50 --L 500 --grad_clip 100 --ep_mcmc_step_size 1e-4 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --case_name case57 --failure_level 6.0 --num_rounds 50 --L 500 --grad_clip 100 --ep_mcmc_step_size 1e-4 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --case_name case57 --failure_level 6.0 --num_rounds 50 --L 500 --grad_clip 100 --ep_mcmc_step_size 1e-4 --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, python radium/experiments/power_systems/solve_scacopf.py --case_name case57 --failure_level 6.0 --num_rounds 50 --L 500 --grad_clip 100 --ep_mcmc_step_size 1e-4 --seed $seed --quench_rounds 20 &
done

wait;