import jax
import jax.numpy as jnp
import sparsediffax as sd
import numpy as np
import scipy.sparse as sp


def f(x):
    return jnp.sum(jnp.cos(jnp.diff(jnp.sin(x))))


def test_hessian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    H_true = jax.hessian(f)(x)
    S_true = sp.diags(
        [np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1], dtype=bool
    ).todense()
    S = sd.naive_hessian_sparsity(f)(x)
    H = sd.hessian_sparse(f, S)(x)
    assert (S.todense() == S_true).all()
    assert jnp.isclose(H.todense(), H_true).all()
