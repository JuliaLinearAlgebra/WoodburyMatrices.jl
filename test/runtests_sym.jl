using Base.Test
using WoodburyMatrices
using Compat:view

srand(123)
n = 5

for elty in (Float32, Float64, Complex64, Complex128, Int), AMat in (diagm,)

    elty = Float64

    a = rand(n); B = rand(n,2); D = rand(2,2); v = rand(n)

    if elty == Int
        v = rand(1:100, n)
        a = rand(1:100, n)
        B = rand(1:100, 2, n)
        D = rand(1:100, 2, 2)
    else
        v = convert(Vector{elty}, v)
        a = convert(Vector{elty}, a)
        B = convert(Matrix{elty}, B)
        D = convert(Matrix{elty}, D)
    end

    ε = eps(abs2(float(one(elty))))
    A = AMat(a)

    # Woodbury
    for W in (SymWoodbury(A, B, D), SymWoodbury(A, B[:,1][:], 2.))

        F = full(W)
        @test (2*W)*v ≈ 2*(W*v)
        @test W'*v ≈ W*v
        @test (W'W)*v ≈ full(W)*(full(W)*v)
        @test (W*W)*v ≈ full(W)*(full(W)*v)
        @test (W*W')*v ≈ full(W)*(full(W)*v)
        @test W[1:3,1:3]*v[1:3] ≈ full(W)[1:3,1:3]*v[1:3]
        @test sparse(W) ≈ full(W)
        @test W === W'
        @test W*eye(n) ≈ full(W)
        @test W'*eye(n) ≈ full(W)

        Z = randn(n,n)
        @test full(W*Z) ≈ full(W)*Z

        R = rand(n,n)

        for v = (rand(n, 1), view(rand(n,1), 1:n), view(rand(n,2),1:n,1:2))
            @test (2*W)*v ≈ 2*(W*v)
            @test (W*2)*v ≈ 2*(W*v)
            @test (W'W)*v ≈ full(W)*(full(W)*v)
            @test (W*W)*v ≈ full(W)*(full(W)*v)
            @test (W*W')*v ≈ full(W)*(full(W)*v)
            @test W[1:3,1:3]*v[1:3] ≈ full(W)[1:3,1:3]*v[1:3]
            @test full(WoodburyMatrices.conjm(W, R)) ≈ R*full(W)*R'
            @test full((copy(W)'W)*v) ≈ full(W)*(full(W)*v)
            @test full(W + A) ≈ full(W)+full(A)
            @test full(A + W) ≈ full(W)+full(A)
        end

        v = rand(n,1)
        W2 = convert(Woodbury, W)
        @test full(W2) ≈ full(W)

        if elty != Int
            @test inv(W)*v ≈ inv(full(W))*v
            @test W\v ≈ inv(full(W))*v
            @test liftFactor(W)(v) ≈ inv(W)*v
            @test WoodburyMatrices.partialInv(W)[1] ≈ inv(W).B
            @test WoodburyMatrices.partialInv(W)[2] ≈ inv(W).D
            @test det(W) ≈ det(full(W))
        end

    end

end


for elty in (Float32, Float64, Complex64, Complex128, Int)

    elty = Float64

    a1 = rand(n); B1 = rand(n,2); D1 = rand(2,2); v = rand(n)
    a2 = rand(n); B2 = rand(n,2); D2 = rand(2,2);

    if elty == Int
        v = rand(1:100, n)

        a1 = rand(1:100, n)
        B1 = rand(1:100, 2, n)
        D1 = rand(1:100, 2, 2)

        a2 = rand(1:100, n)
        B2 = rand(1:100, 2, n)
        D2 = rand(1:100, 2, 2)
    else
        v = convert(Vector{elty}, v)

        a1 = convert(Vector{elty}, a1)
        B1 = convert(Matrix{elty}, B1)
        D1 = convert(Matrix{elty}, D1)

        a2 = convert(Vector{elty}, a2)
        B2 = convert(Matrix{elty}, B2)
        D2 = convert(Matrix{elty}, D2)
    end

    ε = eps(abs2(float(one(elty))))

    # Woodbury
    A1 = diagm(a1)
    A2 = diagm(a2)

    W1 = SymWoodbury(A1, B1, D1)
    W2 = SymWoodbury(A2, B2, D2)

    W1r = SymWoodbury(A1, B1[:,1][:], 2.)
    W2r = SymWoodbury(A2, B2[:,1][:], 3.)

    for (W1, W2) = ((W1,W2), (W1r, W2), (W1, W2r), (W1r,W2r))
        @test (W1 + W2)*v ≈ (full(W1) + full(W2))*v
        @test (full(W1) + W2)*v ≈ (full(W1) + full(W2))*v
        @test (W1 + 2*diagm(a1))*v ≈ (full(W1) + full(2*diagm(a1)))*v
        @test_throws MethodError W1*W2
    end

end

# Sparse U and D

A = diagm(rand(n))
B = sprandn(n,2,1.)
D = sprandn(2,2,1.)
W = SymWoodbury(A, B, D)
v = randn(n)
V = randn(n,1)

@test size(W) == (n,n)
@test size(W,1) == n
@test size(W,2) == n

@test inv(W)*v ≈ inv(full(W))*v
@test (2*W)*v ≈ 2*(W*v)
@test (W'W)*v ≈ full(W)*(full(W)*v)
@test (W*W)*v ≈ full(W)*(full(W)*v)
@test (W*W')*v ≈ full(W)*(full(W)*v)
@test liftFactor(W)(v) ≈ inv(W)*v

@test inv(W)*V ≈ inv(full(W))*V
@test (2*W)*V ≈ 2*(W*V)
@test (W'W)*V ≈ full(W)*(full(W)*V)
@test (W*W)*V ≈ full(W)*(full(W)*V)
@test (W*W')*V ≈ full(W)*(full(W)*V)

# Mismatched sizes
@test_throws DimensionMismatch SymWoodbury(rand(5,5),rand(5,2),rand(2,3))
@test_throws DimensionMismatch SymWoodbury(rand(5,5),rand(5,2),rand(3,3))
@test_throws DimensionMismatch SymWoodbury(rand(5,5),rand(3),1.)
