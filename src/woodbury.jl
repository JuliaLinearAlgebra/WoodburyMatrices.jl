struct Woodbury{T,AType,UType,VType,CType,CpType} <: AbstractWoodbury{T}
    A::AType
    U::UType
    C::CType
    Cp::CpType
    V::VType
    tmpN1::Vector{T}
    tmpN2::Vector{T}
    tmpk1::Vector{T}
    tmpk2::Vector{T}
end

"""
    W = Woodbury(A, U, C, V)

Represent a matrix `W = A + UCV`.
Equations `Wx = b` will be solved using the
[Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity).

If your main goal is to solve equations, it's often advantageous to supply
`A` as a factorization (e.g., `Woodbury(lu(A), U, C, V)`).

See also [SymWoodbury](@ref).
"""
function Woodbury(A, U::AbstractMatrix, C, V::AbstractMatrix)
    @noinline throwdmm1(U, V, A) = throw(DimensionMismatch("Sizes of U ($(size(U))) and/or V ($(size(V))) are inconsistent with A ($(size(A)))"))
    @noinline throwdmm2(k) = throw(DimensionMismatch("C should be $(k)x$(k)"))

    N = size(A, 1)
    k = size(U, 2)
    if size(A, 2) != N || size(U, 1) != N || size(V, 1) != k || size(V, 2) != N
        throwdmm1(U, V, A)
    end
    if k > 1
        if size(C, 1) != k || size(C, 2) != k
            throwdmm2(k)
        end
    end
    Cp = safeinv(safeinv(C) .+ V*(A\U))
    # temporary space for allocation-free solver (vector RHS only)
    T = typeof(float(zero(eltype(A)) * zero(eltype(U)) * zero(eltype(C)) * zero(eltype(V))))
    tmpN1 = Vector{T}(undef, N)
    tmpN2 = Vector{T}(undef, N)
    tmpk1 = Vector{T}(undef, k)
    tmpk2 = Vector{T}(undef, k)

    Woodbury(A, U, C, Cp, V, tmpN1, tmpN2, tmpk1, tmpk2)
end

Woodbury(A, U::AbstractVector{T}, C, V::AbstractMatrix{T}) where {T} = Woodbury(A, reshape(U, length(U), 1), C, V)

# Woodbury(A, U::AbstractVector, C, V::Adjoint) = Woodbury(A, U, C, Matrix(V))

function show(io::IO, W::Woodbury)
    println(io, "Woodbury factorization:\nA:")
    show(io, MIME("text/plain"), W.A)
    print(io, "\nU:\n")
    Base.print_matrix(IOContext(io, :compact=>true), W.U)
    if isa(W.C, Matrix)
        print(io, "\nC:\n")
        Base.print_matrix(IOContext(io, :compact=>true), W.C)
    else
        print(io, "\nC: ", W.C)
    end
    print(io, "\nV:\n")
    Base.print_matrix(IOContext(io, :compact=>true), W.V)
end

copy(W::Woodbury) = Woodbury(W.A, W.U, W.C, W.V)

function inv(W::AbstractWoodbury)
    A′ = inv(W.A)
    U′ = W.A \ W.U
    C′ = -W.Cp
    V′ = W.V*A′
    return Woodbury(A′, U′, C′, V′)
end

+(W::Woodbury, X::Woodbury)       = Woodbury(W.A + X.A, [W.U X.U],
                                             cat(W.C, X.C; dims=(1,2)), [W.V; X.V])
*(α::Real, W::Woodbury)           = Woodbury(α*W.A, W.U, α*W.C, W.V)
+(W::Woodbury, M::AbstractMatrix) = Woodbury(W.A + M, W.U, W.C, W.V)

adjoint(W::Woodbury) = Woodbury(adjoint(W.A), adjoint(W.V), adjoint(W.C), adjoint(W.U))
transpose(W::Woodbury) = Woodbury(transpose(W.A), transpose(W.V), transpose(W.C), transpose(W.U))
