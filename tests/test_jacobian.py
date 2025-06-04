import jax
import jax.numpy as jnp
import sparsediffax as sd


def f(x):
    return jnp.diff(jnp.cos(x))


def test_forward_jacobian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    J = jax.jacfwd(f)(x)
    S = sd.naive_jacfwd_sparsity(f, x)
    Js = sd.jacfwd_sparse(f, S)(x)
    assert jnp.isclose(Js.todense(), J).all()


def test_reverse_jacobian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    J = jax.jacrev(f)(x)
    S = sd.naive_jacrev_sparsity(f, x)
    Js = sd.jacrev_sparse(f, S)(x)
    assert jnp.isclose(Js.todense(), J).all()
