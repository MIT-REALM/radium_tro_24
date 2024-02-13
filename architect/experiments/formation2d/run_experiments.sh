# Run small experiments
# for seed in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 5 --failure_level 10.0 --seed $seed --disable_stochasticity --no-predict &
#     CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 5 --failure_level 10.0 --seed $seed --disable_stochasticity &
#     CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 5 --failure_level 10.0 --seed $seed --disable_gradients --quench_rounds 0 &
#     CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 5 --failure_level 10.0 --reinforce --seed $seed &
#     CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 5 --failure_level 10.0 --seed $seed &
# done

# wait;

# Run large experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 25 --width 6.4 --height 6.0 --failure_level 10.0 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 25 --width 6.4 --height 6.0 --failure_level 10.0 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 25 --width 6.4 --height 6.0 --failure_level 10.0 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 25 --width 6.4 --height 6.0 --failure_level 10.0 --reinforce --seed $seed &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/formation2d/solve.py --n 25 --width 6.4 --height 6.0 --failure_level 10.0 --seed $seed 
done
