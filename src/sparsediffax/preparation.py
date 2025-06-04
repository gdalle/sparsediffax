import jax
import jax.experimental.sparse as jsp


def naive_jacobian_sparsity(f, x):
    J = jax.jacfwd(f)(x)
    S = jsp.BCOO.fromdense(J != 0)
    return S


def naive_hessian_sparsity(f, x):
    H = jax.hessian(f)(x)
    S = jsp.BCOO.fromdense(H != 0)
    return S
