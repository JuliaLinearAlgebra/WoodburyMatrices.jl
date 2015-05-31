using WoodburyMatrices
using Base.Test

srand(123)
n = 5

d = 1 .+ rand(n)
dl = -rand(n-1)
du = -rand(n-1)
v = randn(n)
B = randn(n,2)

U = randn(n,2)
V = randn(2,n)
C = randn(2,2)

for elty in (Float32, Float64, Complex64, Complex128, Int)
    if elty == Int
        srand(61516384)
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
    @test size(T, 1) == n
    @test size(T) == (n, n)

    # Woodbury
    W = Woodbury(T, U, C, V)
    F = full(W)
    @test_approx_eq W*v F*v
    iFv = F\v
    @test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε # Revisit. Condition number is wrong
    @test abs((det(W) - det(F))/det(F)) <= n*cond(F)*ε # Revisit. Condition number is wrong
    iWv = similar(iFv)
    if elty != Int
        iWv = A_ldiv_B!(W, copy(v))
        @test_approx_eq iWv iFv
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
W = Woodbury(T, U, C, V)
F = full(W)

v = randn(n)
ε = eps()
iFv = F\v
@test norm(W\v - iFv)/norm(iFv) <= n*cond(F)*ε
