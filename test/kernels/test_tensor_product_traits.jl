using FunctionalGPs
using KernelFunctions: ColVecs, with_lengthscale, Matern52Kernel
using LinearAlgebra

@testset "kernel_evaluate_evaluate with KernelTensorProduct" begin
    k₁ = with_lengthscale(Matern52Kernel(), 0.8)
    k₂ = with_lengthscale(Matern52Kernel(), 1.2)
    k = k₁ ⊗ k₂

    @testset "One-arg with vector-of-vectors" begin
        X = [[0.1, 0.2], [0.3, 0.5], [0.7, 0.9], [0.4, 0.6]]
        K = kernel_evaluate_evaluate(k, X)

        X_mat = reduce(hcat, X)
        K_ref = kernelmatrix(k, ColVecs(X_mat))
        @test K ≈ K_ref
    end

    @testset "Two-arg with vector-of-vectors" begin
        X_left = [[0.1, 0.2], [0.3, 0.5], [0.7, 0.9]]
        X_right = [[0.15, 0.25], [0.4, 0.6]]
        K = kernel_evaluate_evaluate(k, X_left, X_right)

        X_l_mat = reduce(hcat, X_left)
        X_r_mat = reduce(hcat, X_right)
        K_ref = kernelmatrix(k, ColVecs(X_l_mat), ColVecs(X_r_mat))
        @test K ≈ K_ref
    end
end
