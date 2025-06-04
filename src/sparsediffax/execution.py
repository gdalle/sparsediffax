import scipy.sparse as sp
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import pysparsematrixcolorings as smc


def _jacfwd_compressed(f, x, basis_matrix):
    def _jvp(s):
        return jax.jvp(f, (x,), (s,))[1]

    Jt = jax.vmap(_jvp, in_axes=1)(basis_matrix)
    return jnp.transpose(Jt)


def _jacrev_compressed(f, x, basis_matrix):
    y, vjp_fun = jax.vjp(f, x)
    (J,) = jax.vmap(vjp_fun, in_axes=0)(basis_matrix)
    return J


class SparseDiffPreparation:
    def __init__(
        self,
        S,
        structure: str = "nonsymmetric",
        partition: str = "column",
        order: str = "natural",
    ):
        result = smc.compute_coloring(
            S, structure=structure, partition=partition, order=order, return_aux=True
        )
        colors, basis_matrix, (row_inds_scipy_csc, col_inds_scipy_csc) = result

        row_inds_scipy_coo = sp.coo_matrix(row_inds_scipy_csc)
        col_inds_scipy_coo = sp.coo_matrix(col_inds_scipy_csc)

        rows = row_inds_scipy_coo.row
        cols = row_inds_scipy_coo.col
        row_inds = row_inds_scipy_coo.data
        col_inds = col_inds_scipy_coo.data

        self.shape = S.shape
        self.partition = partition
        self.colors = jnp.array(colors)
        self.basis_matrix = jnp.array(basis_matrix)
        self.indices = jnp.transpose(jnp.stack([rows, cols]))
        self.row_inds = jnp.array(row_inds)
        self.col_inds = jnp.array(col_inds)

    def compressed_jacobian(self, f, x):
        if self.partition == "column":
            return _jacfwd_compressed(f, x, self.basis_matrix.astype(x.dtype))
        else:
            return _jacrev_compressed(f, x, self.basis_matrix.astype(x.dtype))

    def decompress(self, B):
        data = B[self.row_inds, self.col_inds]
        return jsp.BCOO((data, self.indices), shape=self.shape)

    def jacobian(self, f, x):
        B = self.compressed_jacobian(f, x)
        J = self.decompress(B)
        return J
