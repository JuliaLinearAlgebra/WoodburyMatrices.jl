# WoodburyMatrices

[![Build Status](https://travis-ci.org/timholy/WoodburyMatrices.jl.svg?branch=master)](https://travis-ci.org/timholy/WoodburyMatrices.jl)

This package provides support for the [Woodbury matrix identity](http://en.wikipedia.org/wiki/Woodbury_matrix_identity) for the Julia programming language.  This is a generalization of the [Sherman-Morrison formula](http://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula).

## Usage

```julia
using WoodburyMatrices
W = Woodbury(A, U, C, V)
```
creates a `Woodbury` matrix from the `A`, `U`, `C`, and `V` matrices.

There are only a few things you can do with a Woodbury matrix:
- `full(W)` converts to its dense representation
- `W\b` solves the equation `W*x = b` for `x`. Note that the Woodbury matrix identity is notorious for floating-point roundoff errors, so be prepared for a certain amount of inaccuracy in the result.
- `det(W)` computes the determinant of `W`.
