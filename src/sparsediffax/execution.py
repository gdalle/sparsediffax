import scipy.sparse as sp
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import pysparsematrixcolorings as smc
from jax import Array
from typing import Callable

# Preparation with coloring


class _SparseDiff:
    def __init__(
        self,
        sparsity_pattern: jsp.BCOO,
        structure: str,
        partition: str,
        order: str = "natural",
    ):
        assert isinstance(sparsity_pattern, jsp.BCOO)

        sparsity_pattern_scipy_coo = sp.coo_matrix(
            (
                sparsity_pattern.data,
                (sparsity_pattern.indices[:, 0], sparsity_pattern.indices[:, 1]),
            ),
            shape=sparsity_pattern.shape,
        )

        result = smc.compute_coloring(
            sparsity_pattern_scipy_coo,
            structure=smc.Structure(structure),
            partition=smc.Partition(partition),
            order=smc.Order(order),
        )
        colors, basis_matrix, (row_inds_scipy_csc, col_inds_scipy_csc) = result

        row_inds_scipy_coo = sp.coo_matrix(row_inds_scipy_csc)
        col_inds_scipy_coo = sp.coo_matrix(col_inds_scipy_csc)

        rows = row_inds_scipy_coo.row
        cols = row_inds_scipy_coo.col
        row_inds = row_inds_scipy_coo.data
        col_inds = col_inds_scipy_coo.data

        self.sparsity_pattern = sparsity_pattern
        self.partition = partition
        self.colors = jnp.array(colors)
        self.basis_matrix = jnp.array(basis_matrix)
        self.indices = jnp.transpose(jnp.stack([rows, cols]))
        self.row_inds = jnp.array(row_inds)
        self.col_inds = jnp.array(col_inds)

    def decompress(self, B: Array):
        data = B[self.row_inds, self.col_inds]
        return jsp.BCOO((data, self.indices), shape=self.sparsity_pattern.shape)


# Compressed differentiation


def _jacfwd_compressed(f: Callable, x: Array, basis_matrix: Array) -> Array:
    def _jvp(s):
        return jax.jvp(f, (x,), (s,))[1]

    Jt = jax.vmap(_jvp, in_axes=1)(basis_matrix)
    return jnp.transpose(Jt)


def _jacrev_compressed(f: Callable, x: Array, basis_matrix: Array) -> Array:
    y, vjp_fun = jax.vjp(f, x)
    (J,) = jax.vmap(vjp_fun, in_axes=0)(basis_matrix)
    return J


def _hessian_compressed(f: Callable, x: Array, basis_matrix: Array) -> Array:
    def _hvp(s):
        return jax.jvp(jax.grad(f), (x,), (s,))[1]

    Jt = jax.vmap(_hvp, in_axes=1)(basis_matrix)
    return jnp.transpose(Jt)


# Full differentiation


def jacfwd_sparse(f: Callable, sparsity_pattern: jsp.BCOO) -> Callable:
    prep = _SparseDiff(sparsity_pattern, structure="nonsymmetric", partition="column")

    def jacfwd_sparse_fun(x):
        B = _jacfwd_compressed(f, x, prep.basis_matrix.astype(x.dtype))
        return prep.decompress(B)

    return jacfwd_sparse_fun


def jacrev_sparse(f: Callable, sparsity_pattern: jsp.BCOO) -> Callable:
    prep = _SparseDiff(sparsity_pattern, structure="nonsymmetric", partition="row")

    def jacrev_sparse_fun(x):
        B = _jacrev_compressed(f, x, prep.basis_matrix.astype(x.dtype))
        return prep.decompress(B)

    return jacrev_sparse_fun


def hessian_sparse(f: Callable, sparsity_pattern: jsp.BCOO) -> Callable:
    prep = _SparseDiff(sparsity_pattern, structure="symmetric", partition="column")

    def hessian_sparse_fun(x):
        B = _hessian_compressed(f, x, prep.basis_matrix.astype(x.dtype))
        return prep.decompress(B)

    return hessian_sparse_fun
