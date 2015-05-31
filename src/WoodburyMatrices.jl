module WoodburyMatrices

import Base: A_ldiv_B!, convert, copy, det, full, show, similar, size

export Woodbury

#### Woodbury matrices ####
# This type provides support for the Woodbury matrix identity
type Woodbury{T,AType,UType,VType,CType,CpType} <: AbstractMatrix{T}
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

function Woodbury{T}(A, U::AbstractMatrix{T}, C::AbstractMatrix{T}, V::AbstractMatrix{T})
    N = size(A, 1)
    k = size(U, 2)
    if size(A, 2) != N || size(U, 1) != N || size(V, 1) != k || size(V, 2) != N
        throw(DimensionMismatch())
    end
    if k > 1
        if size(C, 1) != k || size(C, 2) != k
            throw(DimensionMismatch())
        end
    end
    Cp = inv(inv(C) + V*(A\U))
    # temporary space for allocation-free solver
    tmpN1 = Array(T, N)
    tmpN2 = Array(T, N)
    tmpk1 = Array(T, k)
    tmpk2 = Array(T, k)
    # don't copy A, it could be huge
    Woodbury{T,typeof(A),typeof(U),typeof(V),typeof(C),typeof(Cp)}(A, copy(U), copy(C), Cp, copy(V), tmpN1, tmpN2, tmpk1, tmpk2)
end

Woodbury{T}(A, U::Vector{T}, C, V::Matrix{T}) = Woodbury(A, reshape(U, length(U), 1), C, V)

size(W::Woodbury) = size(W.A)

function show(io::IO, W::Woodbury)
    println(io, summary(W), ":")
    print(io, "A:\n", W.A)
    print(io, "\nU:\n")
    Base.print_matrix(io, W.U)
    if isa(W.C, Matrix)
        print(io, "\nC:\n")
        Base.print_matrix(io, W.C)
    else
        print(io, "\nC: ", W.C)
    end
    print(io, "\nV:\n")
    Base.print_matrix(io, W.V)
end

full{T}(W::Woodbury{T}) = convert(Matrix{T}, W)
convert{T}(::Type{Matrix{T}}, W::Woodbury{T}) = full(W.A) + W.U*W.C*W.V

copy(W::Woodbury) = Woodbury(W.A, W.U, W.C, W.V)

## Woodbury matrix routines ##

*(W::Woodbury, B::StridedVecOrMat)=W.A*B + W.U*(W.C*(W.V*B))

function \(W::Woodbury, R::StridedVecOrMat)
    AinvR = W.A\R
    AinvR - W.A\(W.U*(W.Cp*(W.V*AinvR)))
end

det(W::Woodbury)=det(W.A)*det(W.C)/det(W.Cp)

function A_ldiv_B!(W::Woodbury, B::AbstractVector)
    length(B) == size(W, 1) || throw(DimensionMismatch("Vector length $(length(B)) must match matrix size $(size(W,1))"))
    copy!(W.tmpN1, B)
    Alu = lufact(W.A) # Note. This makes an allocation (unless A::LU). Alternative is to destroy W.A.
    A_ldiv_B!(Alu, W.tmpN1)
    A_mul_B!(W.tmpk1, W.V, W.tmpN1)
    A_mul_B!(W.tmpk2, W.Cp, W.tmpk1)
    A_mul_B!(W.tmpN2, W.U, W.tmpk2)
    A_ldiv_B!(Alu, W.tmpN2)
    for i = 1:length(W.tmpN2)
        @inbounds B[i] = W.tmpN1[i] - W.tmpN2[i]
    end
    B
end

end
