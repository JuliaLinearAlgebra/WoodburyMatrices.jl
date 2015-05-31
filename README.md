# WoodburyMatrices

[![Build Status](https://travis-ci.org/timholy/WoodburyMatrices.jl.svg?branch=master)](https://travis-ci.org/timholy/WoodburyMatrices.jl)
[![Coverage Status](https://coveralls.io/repos/timholy/WoodburyMatrices.jl/badge.svg)](https://coveralls.io/r/timholy/WoodburyMatrices.jl)

This package provides support for the [Woodbury matrix identity](http://en.wikipedia.org/wiki/Woodbury_matrix_identity) for the Julia programming language.  This is a generalization of the [Sherman-Morrison formula](http://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula).

## Usage

```julia
using WoodburyMatrices
W = Woodbury(A, U, C, V)
```
creates a `Woodbury` matrix from the `A`, `U`, `C`, and `V` matrices.
These matrices can be dense or sparse (or generally any type of `AbstractMatrix`), with the caveat that
`inv(inv(C) + V*(A\U))` will be calculated explicitly and hence needs to be representable with the available resources.
(In many applications, this is a fairly small matrix.)

There are only a few things you can do with a Woodbury matrix:
- `full(W)` converts to its dense representation
- `W\b` solves the equation `W*x = b` for `x`. Note that the Woodbury matrix identity is notorious for floating-point roundoff errors, so be prepared for a certain amount of inaccuracy in the result.
- `det(W)` computes the determinant of `W`.

It's worth emphasizing that `A` can be supplied as a factorization, which makes `W\b` and `det(W)` more efficient.
