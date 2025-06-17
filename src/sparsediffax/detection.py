import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
from typing import Callable


def naive_jacfwd_sparsity(f: Callable) -> Callable:
    def naive_jacfwd_sparsity_fun(x):
        def _jvp(s):
            return jax.jvp(f, (x,), (s,))[1]

        basis_nan = jnp.diag(jnp.full(len(x), jnp.nan))
        Jt = jax.vmap(_jvp, in_axes=1)(basis_nan)
        J = jnp.transpose(Jt)
        return jsp.BCOO.fromdense(jnp.isnan(J))

    return naive_jacfwd_sparsity_fun


def naive_jacrev_sparsity(f: Callable) -> Callable:
    def naive_jacrev_sparsity_fun(x):
        y, vjp_fun = jax.vjp(f, x)
        basis_nan = jnp.diag(jnp.full(len(y), jnp.nan))
        (J,) = jax.vmap(vjp_fun, in_axes=0)(basis_nan)
        return jsp.BCOO.fromdense(jnp.isnan(J))

    return naive_jacrev_sparsity_fun


def naive_hessian_sparsity(f: Callable) -> Callable:
    def naive_hessian_sparsity_fun(x):
        def _hvp(s):
            return jax.jvp(jax.grad(f), (x,), (s,))[1]

        basis_nan = jnp.diag(jnp.full(len(x), jnp.nan))
        H = jax.vmap(_hvp, in_axes=1)(basis_nan)
        return jsp.BCOO.fromdense(jnp.isnan(H))

    return naive_hessian_sparsity_fun
