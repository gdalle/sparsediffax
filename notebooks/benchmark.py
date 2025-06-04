# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax==0.6.1",
#     "marimo",
#     "sparsediffax==0.1.0",
#     "sparsejac==0.2.0",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Sparse autodiff in JAX""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import timeit

    import jax
    import jax.numpy as jnp

    import sparsediffax
    import sparsejac
    return jax, jnp, sparsediffax, sparsejac, timeit


@app.cell
def _():
    samples_nojit = 10
    samples_jit = 100
    return samples_jit, samples_nojit


@app.cell
def _(mo):
    mo.md(r"""## Jacobians""")
    return


@app.cell
def _(jnp):
    def f_vec(x):
        return jnp.cos(jnp.diff(x))
    return (f_vec,)


@app.cell
def _(jax):
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(10000,))
    return (x,)


@app.cell
def _(f_vec, sparsediffax, x):
    jac_sparsity = sparsediffax.naive_jacfwd_sparsity(f_vec, x)
    return (jac_sparsity,)


@app.cell
def _(f_vec, jax):
    J0_nojit = jax.jacfwd(f_vec)
    return (J0_nojit,)


@app.cell
def _(f_vec, jac_sparsity, sparsejac):
    J1_nojit = sparsejac.jacfwd(f_vec, jac_sparsity)
    return (J1_nojit,)


@app.cell
def _(f_vec, jac_sparsity, sparsediffax):
    J2_nojit = sparsediffax.jacfwd_sparse(f_vec, jac_sparsity)
    return (J2_nojit,)


@app.cell
def _(J0_nojit, jax):
    J0 = jax.jit(J0_nojit)
    return (J0,)


@app.cell
def _(J1_nojit, jax):
    J1 = jax.jit(J1_nojit)
    return (J1,)


@app.cell
def _(J2_nojit, jax):
    J2 = jax.jit(J2_nojit)
    return (J2,)


@app.cell
def _(J0, J1, jnp, x):
    jnp.all(J1(x).todense() == J0(x))
    return


@app.cell
def _(J0, J2, jnp, x):
    jnp.all(J2(x).todense() == J0(x))
    return


@app.cell
def _(J0_nojit, J1_nojit, J2_nojit, samples_nojit, timeit, x):
    J0_nojit, J1_nojit, J2_nojit, x
    _t0 = (
        min(
            timeit.repeat(
                "J0_nojit(x).block_until_ready()",
                globals=globals(),
                number=samples_nojit,
            )
        )
    )
    _t1 = (
        min(
            timeit.repeat(
                "J1_nojit(x).block_until_ready()",
                globals=globals(),
                number=samples_nojit,
            )
        )
    )
    _t2 = (
        min(
            timeit.repeat(
                "J2_nojit(x).block_until_ready()",
                globals=globals(),
                number=samples_nojit,
            )
        )
    )
    _t0 / _t1, _t0 / _t2
    return


@app.cell
def _(J0, J1, J2, samples_jit, timeit, x):
    J0, J1, J2, x
    _t0 = (
        min(
            timeit.repeat(
                "J0(x).block_until_ready()", globals=globals(), number=samples_jit
            )
        )
    )
    _t1 = (
        min(
            timeit.repeat(
                "J1(x).block_until_ready()", globals=globals(), number=samples_jit
            )
        )
    )
    _t2 = (
        min(
            timeit.repeat(
                "J2(x).block_until_ready()", globals=globals(), number=samples_jit
            )
        )
    )
    _t0 / _t1, _t0 / _t2
    return


@app.cell
def _(mo):
    mo.md(r"""## Hessians""")
    return


@app.cell
def _(jnp):
    def f_num(x):
        return jnp.sum(jnp.cos(jnp.diff(x))) + jnp.square(x[0]) * jnp.sum(x)
    return (f_num,)


@app.cell
def _(f_num, sparsediffax, x):
    hess_sparsity = sparsediffax.naive_hessian_sparsity(f_num, x)
    return (hess_sparsity,)


@app.cell
def _(hess_sparsity):
    hess_sparsity.todense()
    return


@app.cell
def _(f_num, jax):
    H0_nojit = jax.hessian(f_num)
    return (H0_nojit,)


@app.cell
def _(f_num, hess_sparsity, jax, sparsejac):
    H1_nojit = sparsejac.jacfwd(jax.grad(f_num), hess_sparsity)
    return (H1_nojit,)


@app.cell
def _(f_num, hess_sparsity, sparsediffax):
    H2_nojit = sparsediffax.hessian_sparse(f_num, hess_sparsity)
    return (H2_nojit,)


@app.cell
def _(H0_nojit, jax):
    H0 = jax.jit(H0_nojit)
    return (H0,)


@app.cell
def _(H1_nojit, jax):
    H1 = jax.jit(H1_nojit)
    return (H1,)


@app.cell
def _(H2_nojit, jax):
    H2 = jax.jit(H2_nojit)
    return (H2,)


@app.cell
def _(H0, H1, jnp, x):
    jnp.all(H1(x).todense() == H0(x))
    return


@app.cell
def _(H0, H2, jnp, x):
    jnp.all(H2(x).todense() == H0(x))
    return


@app.cell
def _(H0_nojit, H1_nojit, H2_nojit, samples_nojit, timeit, x):
    H0_nojit, H1_nojit, H2_nojit, x
    _t0 = min(
        timeit.repeat(
            "H0_nojit(x).block_until_ready()",
            globals=globals(),
            number=samples_nojit,
        )
    )
    _t1 = min(
        timeit.repeat(
            "H1_nojit(x).block_until_ready()",
            globals=globals(),
            number=samples_nojit,
        )
    )
    _t2 = min(
        timeit.repeat(
            "H2_nojit(x).block_until_ready()",
            globals=globals(),
            number=samples_nojit,
        )
    )
    _t0 / _t1, _t0 / _t2
    return


@app.cell
def _(H0, H1, H2, samples_jit, timeit, x):
    H0, H1, H2, x
    _t0 = min(
        timeit.repeat(
            "H0(x).block_until_ready()", globals=globals(), number=samples_jit
        )
    )
    _t1 = min(
        timeit.repeat(
            "H1(x).block_until_ready()", globals=globals(), number=samples_jit
        )
    )
    _t2 = min(
        timeit.repeat(
            "H2(x).block_until_ready()", globals=globals(), number=samples_jit
        )
    )
    _t0 / _t1, _t0 / _t2
    return


if __name__ == "__main__":
    app.run()
