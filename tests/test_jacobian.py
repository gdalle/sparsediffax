import jax
import jax.numpy as jnp
import sparsediffax as sd


def f(x):
    return jnp.diff(jnp.cos(x))


def test_jacobian():
    n = 5
    x = jnp.arange(1, n + 1, dtype=float)
    J = jax.jacfwd(f)(x)
    S = sd.naive_jacobian_sparsity(f, x)

    sparsejac = sd.SparseJacobian(S, direction="forward")
    Js = sparsejac.evaluate(f, x)
    assert jnp.all(Js.todense() - J == 0)
