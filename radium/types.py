"""Define some useful types."""
from beartype.typing import Callable
from jaxtyping import Array, Float, Integer, PyTree

PRNGKeyArray = Integer[Array, "2"]
Params = PyTree[Float[Array, "..."]]
LogLikelihood = Callable[[Params], Float[Array, ""]]
Sampler = Callable[[PRNGKeyArray, PyTree], PyTree]
