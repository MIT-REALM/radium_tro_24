"""Code to stress test a design against variation in the exogenous parameters."""
import os

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pandas as pd
from beartype.typing import Callable
from jaxtyping import Array, Float
from tqdm import tqdm

from radium.types import Params, PRNGKeyArray

# Convenience type for cost functions
CostFn = jax.typing.Callable[[Params, Params], Float[Array, ""]]


def stress_test(
    cost_function: CostFn,
    design_params: Params,
    exogenous_sampler: Callable[[PRNGKeyArray], Params],
    algorithm_name: str,
    N: int = 100_000,
    batches: int = 10,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Test the given design against a large number of random exogenous parameters.

    Args:
        cost_function: The cost function to evaluate (taking design and exogenous parameters)
        design_params: The parameters of the design to test
        exogenous_sampler: A function that samples random exogenous parameters
        algorithm_name: The display name of the algorithm that generated this design
        N: The number of exogenous parameters to sample in each batch
        batches: The number of batches to sample
        save_path: The path to save the results to (as a .csv file). Does not save if None.

    Returns:
        The worst-case cost function at N randomly sampled exogenous parameters, returned
        as a Pandas DataFrame
    """
    # Set up a random number generator key to use for random sampling
    key = jrandom.PRNGKey(0)

    # Test the quality of the solution (and the predicted failure modes)
    # by sampling a whole heckin' bunch of exogenous parameters

    # Sample the exogenous parameters in batches
    costs = pd.Series([], dtype=float)
    for _ in tqdm(range(batches)):
        # Sample the exogenous parameters
        key, subkey = jrandom.split(key)
        subkeys = jrandom.split(subkey, N)
        exogenous_params = jax.vmap(exogenous_sampler)(prng_key)

        # Evaluate the cost function at the exogenous parameters
        costs = jax.jit(jax.vmap(cost_function, in_axes=(None, 0)))(
            design_params, exogenous_params
        )

        # Accumulate the results
        costs = pd.concat([costs, pd.Series(costs)], ignore_index=True)

    # Make a dataframe with the results
    results = pd.DataFrame(
        {
            "Algorithm": algorithm_name,
            "Cost": costs,
            "Type": "Observed",
        }
    )

    # Save to csv if requested (make the directory if it doesn't exist)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), parents=True, exist_ok=True)
        results.to_csv(save_path)

    return results
