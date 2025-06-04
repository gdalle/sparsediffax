r"""
Prototype for sparse automatic differentiation in JAX + Julia
"""

from .detection import (
    naive_jacfwd_sparsity,
    naive_jacrev_sparsity,
    naive_hessian_sparsity,
)
from .execution import jacfwd_sparse, jacrev_sparse, hessian_sparse

__all__ = [
    "naive_jacfwd_sparsity",
    "naive_jacrev_sparsity",
    "naive_hessian_sparsity",
    "jacfwd_sparse",
    "jacrev_sparse",
    "hessian_sparse",
]
