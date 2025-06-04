import jax
import jax.experimental.sparse as jsp
from jax import Array
from typing import Callable


def naive_jacfwd_sparsity(f: Callable, x: Array) -> jsp.BCOO:
    J = jax.jacfwd(f)(x)
    S = jsp.BCOO.fromdense(J != 0)
    return S


def naive_jacrev_sparsity(f: Callable, x: Array) -> jsp.BCOO:
    J = jax.jacrev(f)(x)
    S = jsp.BCOO.fromdense(J != 0)
    return S


def naive_hessian_sparsity(f: Callable, x: Array) -> jsp.BCOO:
    H = jax.hessian(f)(x)
    S = jsp.BCOO.fromdense(H != 0)
    return S
