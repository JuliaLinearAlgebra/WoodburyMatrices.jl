import Base:+,*,-,\,^,sparse,full,copy

using Base.LinAlg.BLAS:gemm!,gemm

"""
Represents a matrix of the form A + BDBᵀ.
"""
type SymWoodbury{T,AType, BType, DType} <: AbstractMatrix{T}
  A::AType; 
  B::BType; 
  D::DType;
end

"""
    SymWoodbury(A, B, D)

Represents a matrix of the form A + BDBᵀ.
"""
function SymWoodbury{T}(A, B::AbstractMatrix{T}, D)
    n = size(A, 1)
    k = size(B, 2)
    if size(A, 2) != n || size(B, 1) != n || size(D,1) != k || size(D,2) != k
        throw(DimensionMismatch("Sizes of B ($(size(B))) and/or D ($(size(D))) are inconsistent with A ($(size(A)))"))
    end
    SymWoodbury{T, typeof(A),typeof(B),typeof(D)}(A,B,D)
end

convert{W<:Woodbury}(::Type{W}, O::SymWoodbury) = Woodbury(O.A, O.B, O.D, O.B')

function calc_inv(A, B, D)
  W = inv(A);
  X = W*B;
  invD = -inv(D);
  Z = inv(invD - B'*X);
  SymWoodbury(W,X,Z);
end

Base.inv{T<:Any, AType<:Any, BType<:Any, DType<:AbstractMatrix}(O::SymWoodbury{T,AType,BType,DType}) = 
  calc_inv(O.A, O.B, O.D)

# D is typically small, so this is acceptable. 
Base.inv{T<:Any, AType<:Any, BType<:Any, DType<:SparseMatrixCSC}(O::SymWoodbury{T,AType,BType,DType}) = 
  calc_inv(O.A, O.B, full(O.D));

\(W::SymWoodbury, R::StridedVecOrMat) = inv(W)*R

"""
    partialInv(O)

Get the factors (X,Z) in W + XZXᵀ where W + XZXᵀ = inv( A + BDBᵀ )
"""
function partialInv(O::SymWoodbury)
  X = (O.A)\O.B;
  invD = -1*inv(O.D);
  Z = inv(invD - O.B'*X);
  return (X,Z);
end

function liftFactorVars(A,B,D)
  A  = sparse(A)
  B  = sparse(B)
  Di = sparse(inv(D))
  n  = size(A,1)
  k  = size(B,2)
  M = [A    B   ;
       B'  -Di ];
  M = lufact(M) # ldltfact once it's available.
  return x -> (M\[x; zeros(k,1)])[1:n,:];
end

function liftFactorVars(A,B,D::SparseMatrixCSC)
  liftFactorVars(A,B,full(D))
end

"""
    liftFactor(A)

More stable version of inv(A).  Returns a function which computs the inverse
on evaluation, i.e. `liftFactor(A)(x)` is the same as `inv(A)*x`.
"""
liftFactor(O::SymWoodbury) = liftFactorVars(O.A,O.B,O.D)

function *{T}(O::SymWoodbury{T}, x::Union{Matrix,Vector,SubArray}) 
  o = O.A*x;
  plusBDBtx!(o, O.B, O.D, x)
  return o
end

function plusBDBtx!(o, B, D, x)
	o[:] = o + B*(D*(B'x))
end

# Optimization - use specialized BLAS package 
function plusBDBtx!(o, B::Array{Float64,2}, D, x::Array{Float64,2})
  w = D*gemm('T','N',B,x);
  gemm!('N','N',1.,B,w,1., o)
end

Base.Ac_mul_B{T}(O1::SymWoodbury{T}, x::AbstractVector{T}) = O1*x
Base.Ac_mul_B{T}(O1::SymWoodbury{T}, x::AbstractMatrix{T}) = O1*x

+(O::SymWoodbury, M::SymWoodbury)    = SymWoodbury(O.A + M.A, [O.B M.B],
                                                   cat([1,2],O.D,M.D) );
*(α::Real, O::SymWoodbury)           = SymWoodbury(α*O.A, O.B, α*O.D);
*(O::SymWoodbury, α::Real)           = SymWoodbury(α*O.A, O.B, α*O.D);
+(M::AbstractMatrix, O::SymWoodbury) = SymWoodbury(O.A + M, O.B, O.D);
+(O::SymWoodbury, M::AbstractMatrix) = SymWoodbury(O.A + M, O.B, O.D);
Base.size(M::SymWoodbury)            = size(M.A);
Base.size(M::SymWoodbury, i)         = (i == 1 || i == 2) ? size(M)[1] : 1

Base.full{T}(O::SymWoodbury{T})      = full(O.A) + O.B*O.D*O.B'
Base.copy{T}(O::SymWoodbury{T})      = SymWoodbury(copy(O.A), copy(O.B), copy(O.D))

function square(O::SymWoodbury)
  A = O.A^2
  D = O.D
  B = O.B
  Z = [(O.A*B + B) (O.A*B - B) B]
  SymWoodbury(A, Z, full( cat([1,2],D/2,-D/2, D*B'*B*D) ) )
end

"""
The product of two SymWoodbury matrices will generally be a Woodbury Matrix,
except when they are the same, i.e. the user writes A'A or A*A' or A*A.

Z(A + B*D*Bᵀ) = ZA + ZB*D*Bᵀ

This package will not support support left multiplication by a generic 
matrix, to keep return types consistent.
"""
function *(O1::SymWoodbury, O2::SymWoodbury)
  if (O1 === O2) 
    return square(O1)
  else
    if O1.A == O2.A && O1.B == O2.B && O1.D == O2.D
      return square(O1)
    else
      throw(MethodError("The product of two nonidentical SymWoodbury matrices is not necessarily SymWoodbury."))
    end
  end
end

Base.Ac_mul_B(O1::SymWoodbury, O2::SymWoodbury) = O1*O2
Base.A_mul_Bc(O1::SymWoodbury, O2::SymWoodbury) = O1*O2

conjm(O::SymWoodbury, M) = SymWoodbury(M*O.A*M', M*O.B, O.D);

Base.getindex(O::SymWoodbury, I::UnitRange, I2::UnitRange) =
  SymWoodbury(O.A[I,I], O.B[I,:], O.D);

# This is a slow hack, but generally these matrices aren't sparse.
Base.sparse(O::SymWoodbury) = sparse(full(O))

# returns a pointer to the original matrix, this is consistent with the
# behavior of Symmetric in Base.
Base.ctranspose(O::SymWoodbury) = O