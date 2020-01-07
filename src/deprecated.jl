export liftFactor

function liftFactorVars(A,B,D)
    Base.depwarn("liftFactor and liftFactorVars are no longer necessary and will be removed in a future release. Use `O\\x` instead.", :liftFactorVars)
    A  = sparse(A)
    B  = sparse(B)
    Di = sparse(inv(D))
    n  = size(A,1)
    k  = size(B,2)
    M = [A    B   ;
         B'  -Di ];
    M = lu(M) # ldltfact once it's available.
    return x -> (M\[x; zeros(k,1)])[1:n,:]
end

function liftFactorVars(A,B,D::SparseMatrixCSC)
  liftFactorVars(A,B,Matrix(D))
end

# This could be optimized to avoid the extra allocation of a single element matrix
function liftFactorVars(A,B,D::Real)
  liftFactorVars(A,B,ones(1,1)*D)
end

"""
    liftFactor(A)

More stable version of inv(A).  Returns a function which computes the inverse
on evaluation, i.e. `liftFactor(A)(x)` is the same as `inv(A)*x`.
"""
liftFactor(O::SymWoodbury) = liftFactorVars(O.A,O.B,O.D)
