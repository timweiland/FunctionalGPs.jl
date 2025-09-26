@testset "Unified Cholesky" begin
    using ReTest
    using GaussPDE
    using LinearAlgebra
    using SparseArrays
    using Random: seed!
    seed!(4570903245)

    @testset "Dense matrix; N = $(N)" for N in [3, 5, 10]
        A = rand(N, N)
        A = A' * A
        C = cholesky(A)
        L_wrapped = matrix_sqrt(C)
        I_mat = Matrix(I, N, N)
        @test C.L ≈ Matrix(L_wrapped)
    end

    SPARSITY = 0.3
    @testset "Sparse matrix; N = $(N)" for N in [3, 5, 10]
        A = sprand(N, N, SPARSITY)
        A[diagind(A)] = rand(N)
        A = A' * A
        C = cholesky(A)
        L_wrapped = matrix_sqrt(C)
        L = sparse(C.L)
        Pt = invperm(C.p)
        L = L[Pt, :]
        @test L ≈ Matrix(L_wrapped)
    end
end
