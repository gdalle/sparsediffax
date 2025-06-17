import jax
import jax.numpy as jnp
import sparsediffax as sd
import numpy as np
import scipy.sparse as sp


def f(x):
    return jnp.diff(jnp.cos(x))


def test_forward_jacobian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    J_true = jax.jacfwd(f)(x)
    S_true = sp.diags_array(
        [np.ones(n - 1), np.ones(n - 1)], offsets=[0, 1], shape=(n - 1, n), dtype=bool
    ).todense()
    S = sd.naive_jacfwd_sparsity(f)(x)
    J = sd.jacfwd_sparse(f, S)(x)
    assert (S.todense() == S_true).all()
    assert jnp.isclose(J.todense(), J_true).all()


def test_reverse_jacobian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    J_true = jax.jacrev(f)(x)
    S_true = sp.diags_array(
        [np.ones(n - 1), np.ones(n - 1)], offsets=[0, 1], shape=(n - 1, n), dtype=bool
    ).todense()
    S = sd.naive_jacrev_sparsity(f)(x)
    J = sd.jacrev_sparse(f, S)(x)
    assert (S.todense() == S_true).all()
    assert jnp.isclose(J.todense(), J_true).all()
