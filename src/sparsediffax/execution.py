import scipy.sparse as sp
import jax
from jax import Array
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import pysparsematrixcolorings as smc


def _jacfwd_compressed(f, x: Array, basis_matrix: Array) -> Array:
    def _jvp(s):
        return jax.jvp(f, (x,), (s,))[1]

    Jt = jax.vmap(_jvp, in_axes=1)(basis_matrix)
    return jnp.transpose(Jt)


def _jacrev_compressed(f, x: Array, basis_matrix: Array) -> Array:
    y, vjp_fun = jax.vjp(f, x)
    (J,) = jax.vmap(vjp_fun, in_axes=0)(basis_matrix)
    return J


def _hessian_compressed(f, x: Array, basis_matrix: Array) -> Array:
    def _hvp(s):
        return jax.jvp(jax.grad(f), (x,), (s,))[1]

    Jt = jax.vmap(_hvp, in_axes=1)(basis_matrix)
    return jnp.transpose(Jt)


class SparseDiff:
    def __init__(
        self,
        sparsity_pattern,
        structure: str = "nonsymmetric",
        partition: str = "column",
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
            structure=structure,
            partition=partition,
            order=order,
            return_aux=True,
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

    def evaluate(self, f, x: Array):
        B = self.evaluate_compressed(f, x)
        J = self.decompress(B)
        return J


class SparseJacobian(SparseDiff):
    def __init__(
        self, sparsity_pattern, direction: str = "forward", order: str = "natural"
    ):
        super().__init__(
            sparsity_pattern,
            structure="nonsymmetric",
            partition="column" if direction == "forward" else "row",
            order=order,
        )
        self.direction = direction

    def evaluate_compressed(self, f, x: Array) -> Array:
        if self.partition == "column":
            return _jacfwd_compressed(f, x, self.basis_matrix.astype(x.dtype))
        else:
            return _jacrev_compressed(f, x, self.basis_matrix.astype(x.dtype))


class SparseHessian(SparseDiff):
    def __init__(self, sparsity_pattern, order: str = "natural"):
        super().__init__(
            sparsity_pattern,
            structure="symmetric",
            partition="column",
            order=order,
        )

    def evaluate_compressed(self, f, x: Array) -> Array:
        return _hessian_compressed(f, x, self.basis_matrix.astype(x.dtype))
