# sparsediffax

[![CI](https://github.com/gdalle/sparsediffax/workflows/CI/badge.svg)](https://github.com/gdalle/sparsediffax/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/gdalle/sparsediffax/graph/badge.svg?token=PDO4JD3DS1)](https://codecov.io/gh/gdalle/sparsediffax)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://gdalle.github.io/sparsediffax/)

Prototype for sparse automatic differentiation in JAX + Julia.

This package is meant as an alternative to [sparsejac](https://github.com/mfschubert/sparsejac) with the following differences:

- More efficient graph encodings and colorings thanks to the Julia library [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl)
- Optimized symmetry-aware computation of sparse Hessians (although taking the sparse Jacobian of the gradient can also give good results in practice)

See the [documentation](https://gdalle.github.io/sparsediffax) for details on the API.

> [!WARNING]
> This is a work in progress, it needs more docs and tests. Try at your own risk.
