from jax import lax
import jax.numpy as jnp
import sparsediffax as sd


def test_conditional():
    def f1(x):
        return x[0:5]

    def f2(x):
        return x[5:10]

    def f(x):
        return lax.cond(x[0] > 0, f1, f2, x)

    x = jnp.arange(1, 11, dtype=float)
    assert sd.naive_jacfwd_sparsity(f)(x).sum() == 5  # should be 10
