# Run small experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 3 --n_hiders 5 --width 3.2 --height 2.0 --failure_level 0.0 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 3 --n_hiders 5 --width 3.2 --height 2.0 --failure_level 0.0 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 3 --n_hiders 5 --width 3.2 --height 2.0 --failure_level 0.0 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 3 --n_hiders 5 --width 3.2 --height 2.0 --failure_level 0.0 --seed $seed &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 3 --n_hiders 5 --width 3.2 --height 2.0 --failure_level 0.0 --reinforce --seed $seed &
done

wait;

# Run large experiments
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 12 --n_hiders 20 --height 12.8 --width 4.0 --failure_level 0.0 --seed $seed --disable_stochasticity --no-predict &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 12 --n_hiders 20 --height 12.8 --width 4.0 --failure_level 0.0 --seed $seed --disable_stochasticity &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 12 --n_hiders 20 --height 12.8 --width 4.0 --failure_level 0.0 --seed $seed --disable_gradients --quench_rounds 0 &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 12 --n_hiders 20 --height 12.8 --width 4.0 --failure_level 0.0 --seed $seed --quench_rounds 40 &
    CUDA_VISIBLE_DEVICES=, python architect/experiments/hide_and_seek/solve_hide_and_seek.py --num_rounds 100 --n_seekers 12 --n_hiders 20 --height 12.8 --width 4.0 --failure_level 0.0 --reinforce --seed $seed &

    wait;
done
