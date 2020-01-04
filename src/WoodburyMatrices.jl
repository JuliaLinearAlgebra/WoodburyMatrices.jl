module WoodburyMatrices

using LinearAlgebra
import LinearAlgebra: det, ldiv!, mul!, adjoint, transpose
import Base: +, *, \, inv, convert, copy, show, similar, axes, size
using SparseArrays

export AbstractWoodbury, Woodbury, SymWoodbury

abstract type AbstractWoodbury{T} <: Factorization{T} end

safeinv(A) = inv(A)
safeinv(A::SparseMatrixCSC) = safeinv(Matrix(A))

include("woodbury.jl")
include("symwoodbury.jl")
include("sparsefactors.jl")
include("deprecated.jl")

# Traits and algorithms expressible in terms of AbstractWoodbury

size(W::AbstractWoodbury) = size(W.A)
size(W::AbstractWoodbury, d) = size(W.A, d)
axes(W::AbstractWoodbury) = axes(W.A)
axes(W::AbstractWoodbury, d) = axes(W.A, d)

Base.Matrix(W::AbstractWoodbury{T}) where {T} = Matrix{T}(W)
Base.Matrix{T}(W::AbstractWoodbury) where {T} = convert(Matrix{T}, W)
Base.Array(W::AbstractWoodbury) = Matrix(W)
Base.Array{T}(W::AbstractWoodbury) where T = Matrix{T}(W)

convert(::Type{Matrix{T}}, W::AbstractWoodbury) where {T} = convert(Matrix{T}, Matrix(W.A) + W.U*W.C*W.V)

# This is a slow hack, but generally these matrices aren't sparse.
SparseArrays.sparse(W::AbstractWoodbury) = sparse(Matrix(W))

# Multiplication

*(W::AbstractWoodbury, B::AbstractMatrix)=W.A*B + W.U*(W.C*(W.V*B))

function *(W::AbstractWoodbury, x::AbstractVector)
    # A reduced-allocation optimization (using temp storage for the multiplications)
    mul!(W.tmpN1, W.A, x)
    mul!(W.tmpk1, W.V, x)
    mul!(W.tmpk2, W.C, W.tmpk1)
    mul!(W.tmpN2, W.U, W.tmpk2)
    return W.tmpN1 + W.tmpN2
end

# Division

function \(W::AbstractWoodbury, R::AbstractMatrix)
    AinvR = W.A\R
    return AinvR - W.A\(W.U*(W.Cp*(W.V*AinvR)))
end

ldiv!(W::AbstractWoodbury, B::AbstractVector) = ldiv!(B, W, B)

function ldiv!(dest::AbstractVector, W::AbstractWoodbury, B::AbstractVector)
    @noinline throwdmm(W, B) = throw(DimensionMismatch("Vector length $(length(B)) must match matrix size $(size(W,1))"))

    length(B) == size(W, 1) || throwdmm(W, B)
    _ldiv!(dest, W, W.A, B)
    return dest
end

function _ldiv!(dest, W, A::Factorization, B)
    ldiv!(W.tmpN1, A, B)
    mul!(W.tmpk1, W.V, W.tmpN1)
    mul!(W.tmpk2, W.Cp, W.tmpk1)
    mul!(W.tmpN2, W.U, W.tmpk2)
    ldiv!(A, W.tmpN2)
    for i = 1:length(W.tmpN2)
        @inbounds dest[i] = W.tmpN1[i] - W.tmpN2[i]
    end
    return dest
end
_ldiv!(dest, W, A::AbstractMatrix, B) = _ldiv!(dest, W, lu(A), B)

det(W::AbstractWoodbury) = det(W.A)*det(W.C)/det(W.Cp)

*(W::AbstractWoodbury, α::Real) = α*W
+(M::AbstractMatrix, W::AbstractWoodbury) = W + M

end
