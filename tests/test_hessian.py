import jax
import jax.numpy as jnp
import sparsediffax as sd


def f(x):
    return jnp.sum(jnp.cos(jnp.diff(jnp.sin(x))))


def test_hessian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    H = jax.hessian(f)(x)
    S = sd.naive_hessian_sparsity(f, x)
    sparsehess = sd.SparseHessian(S)
    Hs = sparsehess.evaluate(f, x)
    assert jnp.isclose(Hs.todense(), H).all()
