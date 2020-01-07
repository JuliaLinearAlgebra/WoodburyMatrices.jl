@testset "Woodbury" begin
seed!(123)
n = 5

d = 1 .+ rand(n)
dl = -rand(n-1)
du = -rand(n-1)
v = randn(n)
B = randn(n,2)

U = randn(n,2)
V = randn(2,n)
C = randn(2,2)

for elty in (Float32, Float64, ComplexF32, ComplexF64, Int)
    if elty == Int
        seed!(61516384)
        d = rand(1:100, n)
        dl = -rand(0:10, n-1)
        du = -rand(0:10, n-1)
        v = rand(1:100, n)
        B = rand(1:100, n, 2)

        # Woodbury
        U = rand(1:100, n, 2)
        V = rand(1:100, 2, n)
        C = rand(1:100, 2, 2)
    else
        d = convert(Vector{elty}, d)
        dl = convert(Vector{elty}, dl)
        du = convert(Vector{elty}, du)
        v = convert(Vector{elty}, v)
        B = convert(Matrix{elty}, B)
        U = convert(Matrix{elty}, U)
        V = convert(Matrix{elty}, V)
        C = convert(Matrix{elty}, C)
    end
    ε = eps(abs2(float(one(elty))))
    T = Tridiagonal(dl, d, du)

    # Matrix for A
    W = Woodbury(T, U, C, V)
    @test size(W, 1) == n
    @test size(W) == (n, n)
    @test axes(W) === (Base.OneTo(n), Base.OneTo(n))
    @test axes(W, 1) === Base.OneTo(n)
    F = Matrix(W)
    @test F ≈ T + U*C*V
    @test Array(W) == Matrix(W)
    if elty <: Real
        @test eltype(Matrix{Float64}(W)) === Float64
        @test eltype(Array{Float64}(W)) === Float64
    end
    @test W*v ≈ F*v
    iFv = F\v
    @test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε # Revisit. Condition number is wrong
    @test norm(inv(W)*v - iFv)/norm(iFv) <= n*cond(F)*ε
    @test abs((det(W) - det(F))/det(F)) <= n*cond(F)*ε # Revisit. Condition number is wrong
    iWv = similar(iFv)
    if elty != Int
        iWv = ldiv!(W, copy(v))
        @test iWv ≈ iFv
    end
    @test W'*v ≈ F'*v
    @test transpose(W)*v ≈ transpose(F)*v
    @test (W + W) \ v ≈ (2*Matrix(W)) \ v

    # Factorization for A
    W = Woodbury(lu(T), U, C, V)
    F = Matrix(W)
    @test_broken W*v ≈ F*v   # needs mul!(dst, ::LU, v, ...)
    iFv = F\v
    @test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε # Revisit. Condition number is wrong
    @test abs((det(W) - det(F))/det(F)) <= n*cond(F)*ε # Revisit. Condition number is wrong
    iWv = similar(iFv)
    if elty != Int
        iWv = ldiv!(W, copy(v))
        @test iWv ≈ iFv
    end
end

# Sparse U and V
n = 5
d = 1 .+ rand(n)
dl = -rand(n-1)
du = -rand(n-1)
T = Tridiagonal(dl, d, du)
U = sprandn(n,2,0.2)
V = sprandn(2,n,0.2)
C = randn(2,2)
v = randn(n)
ε = eps()

W = Woodbury(T, U, C, V)
F = Matrix(W)
iFv = F\v
@test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε

W = Woodbury(lu(T), U, C, V)
F = Matrix(W)
iFv = F\v
@test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε

# Display
iob = IOBuffer()
show(iob, W)
str = String(take!(iob))
@test occursin("C:", str)

# Vector-valued U and scalar-valued C
U = rand(n)
C = 2.3
V = rand(1,n)
W = Woodbury(T, U, C, V)
F = Matrix(W)

v = randn(n)
ε = eps()
iFv = F\v
@test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε

@test Matrix(2*W) ≈ 2*Matrix(W)
@test Matrix(W*2) ≈ 2*Matrix(W)
R = rand(size(W)...)
@test Matrix(W + R) ≈ Matrix(W) + R
@test Matrix(R + W) ≈ Matrix(W) + R

show(iob, W)

Wc = copy(W)

# Mismatched sizes
@test_throws DimensionMismatch Woodbury(rand(5,5),rand(5,2),rand(2,3),rand(3,5))
@test_throws DimensionMismatch Woodbury(rand(5,5),rand(5,2),rand(3,3),rand(2,5))

# spec builder
n = 10
num_nz = 3
for i in 1:5  # repeat 5 times
    args = [(rand(1:n), rand(1:n), rand()) for j in 1:num_nz]
    r, v, c = WoodburyMatrices.sparse_factors(Float64, n, args...)
    out = r*v*c

    by_hand = zeros(n, n)
    for (i, j, v) in args
        by_hand[i, j] = v
    end

    @test maximum(abs,out - by_hand) == 0.0
end

end  # @testset "Woodbury"
