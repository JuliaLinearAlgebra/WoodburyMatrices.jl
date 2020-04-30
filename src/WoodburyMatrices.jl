module WoodburyMatrices

using LinearAlgebra
import LinearAlgebra: det, logdet, logabsdet, ldiv!, mul!, adjoint, transpose
import Base: +, *, \, inv, convert, copy, show, similar, axes, size
using SparseArrays

export AbstractWoodbury, Woodbury, SymWoodbury

abstract type AbstractWoodbury{T} <: Factorization{T} end

safeinv(A) = inv(A)
safeinv(A::SparseMatrixCSC) = safeinv(Matrix(A))

myldiv!(A, B)       = ldiv!(A, B)
myldiv!(dest, A, B) = ldiv!(dest, A, B)
if VERSION <= v"1.4.0-DEV.635"
    myldiv!(A::Diagonal, B)       = (B .= A.diag .\ B)
    myldiv!(dest, A::Diagonal, B) = (dest .= A.diag .\ B)
end

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

function _ldiv(W::AbstractWoodbury, R::AbstractMatrix)
    AinvR = W.A\R
    return AinvR - W.A\(W.U*(W.Cp*(W.V*AinvR)))
end

\(W::AbstractWoodbury, R::AbstractMatrix) = _ldiv(W, R)
\(W::AbstractWoodbury, D::Diagonal) = _ldiv(W, D)

ldiv!(W::AbstractWoodbury, B::AbstractVector) = ldiv!(B, W, B)

function ldiv!(dest::AbstractVector, W::AbstractWoodbury, B::AbstractVector)
    @noinline throwdmm(W, B) = throw(DimensionMismatch("Vector length $(length(B)) must match matrix size $(size(W,1))"))

    length(B) == size(W, 1) || throwdmm(W, B)
    _ldiv!(dest, W, W.A, B)
    return dest
end

function _ldiv!(dest, W, A::Union{Factorization,Diagonal}, B)
    myldiv!(W.tmpN1, A, B)
    mul!(W.tmpk1, W.V, W.tmpN1)
    mul!(W.tmpk2, W.Cp, W.tmpk1)
    mul!(W.tmpN2, W.U, W.tmpk2)
    myldiv!(A, W.tmpN2)
    for i = 1:length(W.tmpN2)
        @inbounds dest[i] = W.tmpN1[i] - W.tmpN2[i]
    end
    return dest
end
_ldiv!(dest, W, A, B) = _ldiv!(dest, W, lu(A), B)

det(W::AbstractWoodbury) = det(W.A)*det(W.C)/det(W.Cp)
logdet(W::AbstractWoodbury) = logdet(W.A) + logdet(W.C) - logdet(W.Cp)
function logabsdet(W::AbstractWoodbury)
    lad_A = logabsdet(W.A)
    lad_C = logabsdet(W.C)
    lad_Cp = logabsdet(W.Cp)
    lad = lad_A[1] + lad_C[1] - lad_Cp[1]
    s = lad_A[2] * lad_C[2] / lad_Cp[2]
    return lad, s
end

*(W::AbstractWoodbury, α::Real) = α*W
+(M::AbstractMatrix, W::AbstractWoodbury) = W + M

end
