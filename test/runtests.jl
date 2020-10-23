using WoodburyMatrices
using LinearAlgebra, SparseArrays, Test
using Random: seed!

# helper function for testing logdet
function randpsd(sidelength)
    Q = randn(sidelength, sidelength)
    return Q * Q'
end

include("woodbury.jl")
include("symwoodbury.jl")
