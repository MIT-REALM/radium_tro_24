# Run small experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, architect/experiments/power_systems/solve_scacopf.py --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, architect/experiments/power_systems/solve_scacopf.py --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, architect/experiments/power_systems/solve_scacopf.py --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, architect/experiments/power_systems/solve_scacopf.py --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, architect/experiments/power_systems/solve_scacopf.py --seed $seed &
done

wait;

# Run large experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/power_systems/solve_scacopf.py --case_name case57 --seed $seed 
done
