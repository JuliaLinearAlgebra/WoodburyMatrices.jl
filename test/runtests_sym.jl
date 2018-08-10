using WoodburyMatrices
using Test

seed!(123)
n = 5

for elty in (Float32, Float64, ComplexF32, ComplexF64, Int), AMat in (x -> Matrix(Diagonal(x)),)

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

        F = Matrix(W)
        @test (2*W)*v ≈ 2*(W*v)
        @test W'*v ≈ W*v
        @test (W'W)*v ≈ Matrix(W)*(Matrix(W)*v)
        @test (W*W)*v ≈ Matrix(W)*(Matrix(W)*v)
        @test (W*W')*v ≈ Matrix(W)*(Matrix(W)*v)
        @test W[1:3,1:3]*v[1:3] ≈ Matrix(W)[1:3,1:3]*v[1:3]
        @test sparse(W) ≈ Matrix(W)
        @test W === W'
        @test W*Matrix(1.0I, n, n) ≈ Matrix(W)
        @test W'*Matrix(1.0I, n, n) ≈ Matrix(W)

        Z = randn(n,n)
        @test Matrix(W*Z) ≈ Matrix(W)*Z

        R = rand(n,n)

        for v = (rand(n, 1), view(rand(n,1), 1:n), view(rand(n,2),1:n,1:2))
            @test (2*W)*v ≈ 2*(W*v)
            @test (W*2)*v ≈ 2*(W*v)
            @test (W'W)*v ≈ Matrix(W)*(Matrix(W)*v)
            @test (W*W)*v ≈ Matrix(W)*(Matrix(W)*v)
            @test (W*W')*v ≈ Matrix(W)*(Matrix(W)*v)
            @test W[1:3,1:3]*v[1:3] ≈ Matrix(W)[1:3,1:3]*v[1:3]
            @test Matrix(WoodburyMatrices.conjm(W, R)) ≈ R*Matrix(W)*R'
            @test Matrix(copy(W)'W)*v ≈ Matrix(W)*(Matrix(W)*v)
            @test Matrix(W + A) ≈ Matrix(W)+Matrix(A)
            @test Matrix(A + W) ≈ Matrix(W)+Matrix(A)
        end

        v = rand(n,1)
        W2 = convert(Woodbury, W)
        @test Matrix(W2) ≈ Matrix(W)

        if elty != Int
            @test inv(W)*v ≈ inv(Matrix(W))*v
            @test W\v ≈ inv(Matrix(W))*v
            @test liftFactor(W)(v) ≈ inv(W)*v
            @test WoodburyMatrices.partialInv(W)[1] ≈ inv(W).B
            @test WoodburyMatrices.partialInv(W)[2] ≈ inv(W).D
            @test det(W) ≈ det(Matrix(W))
        end

    end

end


for elty in (Float32, Float64, ComplexF32, ComplexF64, Int)

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
    A1 = Matrix(Diagonal((a1)))
    A2 = Matrix(Diagonal((a2)))

    W1 = SymWoodbury(A1, B1, D1)
    W2 = SymWoodbury(A2, B2, D2)

    W1r = SymWoodbury(A1, B1[:,1][:], 2.)
    W2r = SymWoodbury(A2, B2[:,1][:], 3.)

    for (W1, W2) = ((W1,W2), (W1r, W2), (W1, W2r), (W1r,W2r))
        @test (W1 + W2)*v ≈ (Matrix(W1) + Matrix(W2))*v
        @test (Matrix(W1) + W2)*v ≈ (Matrix(W1) + Matrix(W2))*v
        @test (W1 + 2*Matrix(Diagonal((a1))))*v ≈ (Matrix(W1) + Matrix(2*Matrix(Diagonal((a1)))))*v
        @test_throws MethodError W1*W2
    end

end

# Sparse U and D

A = Matrix(Diagonal((rand(n))))
B = sprandn(n,2,1.)
D = sprandn(2,2,1.)
W = SymWoodbury(A, B, D)
v = randn(n)
V = randn(n,1)

@test size(W) == (n,n)
@test size(W,1) == n
@test size(W,2) == n

@test inv(W)*v ≈ inv(Matrix(W))*v
@test (2*W)*v ≈ 2*(W*v)
@test (W'W)*v ≈ Matrix(W)*(Matrix(W)*v)
@test (W*W)*v ≈ Matrix(W)*(Matrix(W)*v)
@test (W*W')*v ≈ Matrix(W)*(Matrix(W)*v)
@test liftFactor(W)(v) ≈ inv(W)*v

@test inv(W)*V ≈ inv(Matrix(W))*V
@test (2*W)*V ≈ 2*(W*V)
@test (W'W)*V ≈ Matrix(W)*(Matrix(W)*V)
@test (W*W)*V ≈ Matrix(W)*(Matrix(W)*V)
@test (W*W')*V ≈ Matrix(W)*(Matrix(W)*V)

# Mismatched sizes
@test_throws DimensionMismatch SymWoodbury(rand(5,5),rand(5,2),rand(2,3))
@test_throws DimensionMismatch SymWoodbury(rand(5,5),rand(5,2),rand(3,3))
@test_throws DimensionMismatch SymWoodbury(rand(5,5),rand(3),1.)
