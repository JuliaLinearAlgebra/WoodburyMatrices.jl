using Base.Test
using WoodburyMatrices

srand(123)
n = 5

for elty in (Float32, Float64, Complex64, Complex128, Int)

    elty = Float64

    a = rand(n); B = rand(n,2); D = rand(2,2); v = rand(n)

    if elty == Int
        v = rand(1:100, n)
        a = rand(1:100, n, 2)
        B = rand(1:100, 2, n)
        D = rand(1:100, 2, 2)
    else
        v = convert(Vector{elty}, a)
        a = convert(Vector{elty}, a)
        B = convert(Matrix{elty}, B)
        D = convert(Matrix{elty}, D)
    end

    ε = eps(abs2(float(one(elty))))
    A = Diagonal(a)
    
    # Woodbury
    W = SymWoodbury(A, B, D)
    F = full(W)

    @test_approx_eq (2*W)*v 2*(W*v)
    @test_approx_eq W'*v W*v
    @test_approx_eq (W'W)*v full(W)*(full(W)*v)
    @test_approx_eq (W*W)*v full(W)*(full(W)*v)
    @test_approx_eq (W*W')*v full(W)*(full(W)*v)
    @test_approx_eq W[1:3,1:3]*v[1:3] full(W)[1:3,1:3]*v[1:3]

    v = rand(n, 1)

    @test_approx_eq (2*W)*v 2*(W*v)
    @test_approx_eq (W'W)*v full(W)*(full(W)*v)
    @test_approx_eq (W*W)*v full(W)*(full(W)*v)
    @test_approx_eq (W*W')*v full(W)*(full(W)*v)
    @test_approx_eq W[1:3,1:3]*v[1:3] full(W)[1:3,1:3]*v[1:3]

    if elty != Int
        @test_approx_eq inv(W)*v inv(full(W))*v
        @test_approx_eq W\v inv(full(W))*v        
        @test_approx_eq liftFactor(W)(v) inv(W)*v
    end

end


for elty in (Float32, Float64, Complex64, Complex128, Int)

    elty = Float64

    a = rand(n); B = rand(n,2); D = rand(2,2); v = rand(n)

    if elty == Int
        v = rand(1:100, n)
        
        a1 = rand(1:100, n, 2)
        B1 = rand(1:100, 2, n)
        D1 = rand(1:100, 2, 2)

        a2 = rand(1:100, n, 2)
        B2 = rand(1:100, 2, n)
        D2 = rand(1:100, 2, 2)
    else
        v = convert(Vector{elty}, a)

        a1 = convert(Vector{elty}, a)
        B1 = convert(Matrix{elty}, B)
        D1 = convert(Matrix{elty}, D)

        a2 = convert(Vector{elty}, a)
        B2 = convert(Matrix{elty}, B)
        D2 = convert(Matrix{elty}, D)
    end

    ε = eps(abs2(float(one(elty))))
    
    # Woodbury
    A1 = Diagonal(a1)
    A2 = Diagonal(a2)

    W1 = SymWoodbury(A1, B1, D1)
    W2 = SymWoodbury(A2, B2, D2)

    @test_approx_eq (W1 + W2)*v (full(W1) + full(W2))*v
    @test_approx_eq (full(W1) + W2)*v (full(W1) + full(W2))*v
    @test_approx_eq (W1 + full(W2))*v (full(W1) + full(W2))*v

end

# Sparse U and D

A = Diagonal(rand(n))
B = sprandn(n,2,1.)
D = sprandn(2,2,1.)
W = SymWoodbury(A, B, D)
v = randn(n)
V = randn(n,1)

@test size(W) == (n,n)
@test size(W,1) == n
@test size(W,2) == n

@test_approx_eq inv(W)*v inv(full(W))*v
@test_approx_eq (2*W)*v 2*(W*v)
@test_approx_eq (W'W)*v full(W)*(full(W)*v)
@test_approx_eq (W*W)*v full(W)*(full(W)*v)
@test_approx_eq (W*W')*v full(W)*(full(W)*v)
@test_approx_eq liftFactor(W)(v) inv(W)*v

@test_approx_eq inv(W)*V inv(full(W))*V
@test_approx_eq (2*W)*V 2*(W*V)
@test_approx_eq (W'W)*V full(W)*(full(W)*V)
@test_approx_eq (W*W)*V full(W)*(full(W)*V)
@test_approx_eq (W*W')*V full(W)*(full(W)*V)

# # Mismatched sizes
# @test_throws DimensionMismatch Woodbury(rand(5,5),rand(5,2),rand(2,3),rand(3,5))
# @test_throws DimensionMismatch Woodbury(rand(5,5),rand(5,2),rand(3,3),rand(2,5))