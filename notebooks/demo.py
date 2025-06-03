# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax==0.6.1",
#     "jaxlib==0.6.1",
#     "juliacall==0.9.25",
#     "juliapkg==0.1.17",
#     "marimo",
#     "numpy==2.2.6",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Sparse autodiff with JAX + Julia

    [Guillaume Dalle](https://gdalle.github.io/)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook showcases a sparse Jacobian computation in JAX + Julia, partly inspired by [`sparsejac`](https://github.com/mfschubert/sparsejac). This package already offers sparse Jacobians with the compression-based approach, but it has several drawbacks:

    - the coloring step is not optimized and does not generalize to sparse Hessians
    - there is no clear separation between the one-time operations that can be amortized and the operations that have to be performed at each differentiation

    See the ICLR 2025 blog post [An Illustrated Guide to Sparse Automatic Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) for more details on the underlying methods.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Import required packages from Python and Julia:""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse
    import juliacall
    import juliapkg
    return jax, jnp, juliapkg, np, sparse


@app.cell
def _(juliapkg):
    from juliacall import Main as jl

    juliapkg.add("SparseMatrixColorings", "0a514795-09f3-496d-8182-132a7b665d35")
    jl.seval("using SparseArrays, SparseMatrixColorings")
    return (jl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define test function and input:""")
    return


@app.cell
def _(jnp):
    def f(x):
        return jnp.cos(x) + jnp.sin(jnp.flip(x))
    return (f,)


@app.cell
def _():
    n = 5
    return (n,)


@app.cell
def _(jnp, n):
    x = jnp.arange(1, n + 1, dtype=float)
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dense Jacobian""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The Jacobian of `f` has an X-like sparsity pattern:""")
    return


@app.cell
def _(f, jax, x):
    jacobian = jax.jacfwd(f)(x)
    return (jacobian,)


@app.cell
def _(jacobian):
    jacobian
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Computing it that way requires 5 JVPs.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Sparsity detection""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we perform a poor man's version of sparsity detection: computing the full dense Jacobian and checking which coefficients are non-zero. This is vulnerable to accidental zeros, unlike symbolic approaches like  [`jax2sympy`](https://github.com/johnviljoen/jax2sympy).""")
    return


@app.cell
def _(jacobian, sparse):
    sparsity_pattern = sparse.BCOO.fromdense(jacobian != 0)
    return (sparsity_pattern,)


@app.cell
def _(sparsity_pattern):
    sparsity_pattern.todense()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In general, sparsity detection might be inefficient, but it only has to be performed once if the pattern is input-agnostic.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Coloring""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Coloring problems and algorithms are available through the Julia package [`SparseMatrixColorings`](https://github.com/gdalle/SparseMatrixColorings.jl). Here we demonstrate the simplest case: column coloring of a nonsymmetric matrix.""")
    return


@app.cell
def _(jl):
    jl_coloring_problem = jl.ColoringProblem(
        structure=jl.Symbol("nonsymmetric"), partition=jl.Symbol("column")
    )
    return (jl_coloring_problem,)


@app.cell
def _(jl):
    jl_coloring_algorithm = jl.GreedyColoringAlgorithm()
    return (jl_coloring_algorithm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Translate the JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html#bcoo-data-structure) matrix into a Julia CSC matrix.""")
    return


@app.cell
def _(jl, np, sparsity_pattern):
    _row_inds = jl.Vector(np.array(sparsity_pattern.indices[:, 0] + 1))
    _col_inds = jl.Vector(np.array(sparsity_pattern.indices[:, 1] + 1))
    _nz_vals = jl.ones(jl.Bool, sparsity_pattern.nse)
    jl_sparsity_pattern = jl.sparse(_row_inds, _col_inds, _nz_vals)
    return (jl_sparsity_pattern,)


@app.cell
def _(jl_sparsity_pattern):
    jl_sparsity_pattern
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Get a vector of 1-indexed colors from the Julia coloring routines:""")
    return


@app.cell
def _(jl, jl_coloring_algorithm, jl_coloring_problem, jl_sparsity_pattern):
    jl_colors = jl.fast_coloring(
        jl_sparsity_pattern, jl_coloring_problem, jl_coloring_algorithm
    )
    return (jl_colors,)


@app.cell
def _(jl_colors):
    jl_colors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Translate it back to a 0-index JAX vector:""")
    return


@app.cell
def _(jl_colors, jnp):
    colors = jnp.array(jl_colors) - 1
    return (colors,)


@app.cell
def _(colors):
    colors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Compression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For each color, sum the basis vectors of all the columns assigned to that color:""")
    return


@app.cell
def _(colors, jax, jnp, n):
    _ncolors = jnp.max(colors) + 1
    _colors_onehot = jax.nn.one_hot(colors, _ncolors, dtype=bool)
    basis_vectors = jnp.column_stack(
        [jnp.eye(n)[:, _colors_onehot[:, c]].sum(axis=1) for c in range(_ncolors)]
    )
    return (basis_vectors,)


@app.cell
def _(basis_vectors):
    basis_vectors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Evaluating JVPs with each of these basis vectors is sufficient to recover every nonzero coefficient:""")
    return


@app.cell
def _(f, jax, jnp):
    def jac_compressed(x, basis_vectors):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        Jt = jax.vmap(_jvp, in_axes=1)(basis_vectors)
        return jnp.transpose(Jt)
    return (jac_compressed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The compressed Jacobian has one column per color, it can be evaluated with 2 JVPs instead of 5:""")
    return


@app.cell
def _(basis_vectors, jac_compressed, x):
    jacobian_compressed = jac_compressed(x, basis_vectors)
    return (jacobian_compressed,)


@app.cell
def _(jacobian_compressed):
    jacobian_compressed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Decompression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Decompression works by associating each entry in the sparsity pattern to the corresponding value inside the compressed Jacobian:""")
    return


@app.cell
def _(colors, jacobian_compressed, n, sparse, sparsity_pattern):
    _compressed_row_indices = sparsity_pattern.indices[:, 0]
    _compressed_column_indices = colors[sparsity_pattern.indices[:, 1]]
    _data = jacobian_compressed[
        _compressed_row_indices, _compressed_column_indices
    ]
    jacobian_sparse = sparse.BCOO((_data, sparsity_pattern.indices), shape=(n, n))
    return (jacobian_sparse,)


@app.cell
def _(jacobian_sparse):
    jacobian_sparse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The result is equal to the dense Jacobian:""")
    return


@app.cell
def _(jacobian, jacobian_sparse):
    jacobian_sparse.todense() - jacobian
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Full pipeline""")
    return


@app.cell
def _(jax, jl, jnp, n, np):
    def prepare_sparse_jacobian(sparsity_pattern):
        # sparsity pattern translation
        _row_inds = jl.Vector(np.array(sparsity_pattern.indices[:, 0] + 1))
        _col_inds = jl.Vector(np.array(sparsity_pattern.indices[:, 1] + 1))
        _nz_vals = jl.ones(jl.Bool, sparsity_pattern.nse)
        jl_sparsity_pattern = jl.sparse(_row_inds, _col_inds, _nz_vals)

        # coloring
        jl_coloring_problem = jl.ColoringProblem(
            structure=jl.Symbol("nonsymmetric"), partition=jl.Symbol("column")
        )
        jl_coloring_algorithm = jl.GreedyColoringAlgorithm()

        jl_colors = jl.fast_coloring(
            jl_sparsity_pattern, jl_coloring_problem, jl_coloring_algorithm
        )
        colors = jnp.array(jl_colors) - 1

        # basis vectors
        _ncolors = jnp.max(colors) + 1
        _colors_onehot = jax.nn.one_hot(colors, _ncolors, dtype=bool)
        basis_vectors = jnp.column_stack(
            [
                jnp.eye(n)[:, _colors_onehot[:, c]].sum(axis=1)
                for c in range(_ncolors)
            ]
        )

        return colors, basis_vectors
    return (prepare_sparse_jacobian,)


@app.cell
def _(jac_compressed, n, sparse):
    def compute_sparse_jacobian(f, x, sparsity_pattern, colors, basis_vectors):
        # differentiation
        jacobian_compressed = jac_compressed(x, basis_vectors)

        # decompression
        _compressed_row_indices = sparsity_pattern.indices[:, 0]
        _compressed_column_indices = colors[sparsity_pattern.indices[:, 1]]
        _data = jacobian_compressed[
            _compressed_row_indices, _compressed_column_indices
        ]
        jacobian_sparse = sparse.BCOO(
            (_data, sparsity_pattern.indices), shape=(n, n)
        )

        return jacobian_sparse
    return (compute_sparse_jacobian,)


@app.cell
def _(
    compute_sparse_jacobian,
    f,
    prepare_sparse_jacobian,
    sparsity_pattern,
    x,
):
    _colors, _basis_vectors = prepare_sparse_jacobian(sparsity_pattern)
    compute_sparse_jacobian(f, x, sparsity_pattern, _colors, _basis_vectors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Benchmarking""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Todo:

    - [ ] Add JIT compilation
    - [ ] Compare to dense Jacobian on large vectors
    """
    )
    return


if __name__ == "__main__":
    app.run()
