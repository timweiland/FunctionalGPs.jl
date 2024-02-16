using Test
using GaussPDE
using KernelFunctions

@testset "Factorized Grid" begin
    k₁ = with_lengthscale(Matern32Kernel(), 1.0)
    k₂ = with_lengthscale(Matern32Kernel(), 0.7)
    k = k₁ ⊗ k₂

    flat_arr(G) = RowVecs(reshape(convert(Array, G), :, 2))

    @testset "KernelTensorProduct k(X, X)" for (N, M) in
                                               Iterators.product([5, 10, 15], [5, 10, 15])
        x₁ = rand(0.0:0.1:3.0, N)
        x₂ = rand(0.0:0.1:3.0, M)
        G = FactorizedGrid(x₁, x₂)

        K = kernelmatrix(k, flat_arr(G))

        @test kernelmatrix(k, G) ≈ K atol = 1e-8
    end

    @testset "KernelTensorProduct k(X, Y)" for (N, M) in
                                               Iterators.product([5, 10, 15], [5, 10, 15])
        x₁ = rand(0.0:0.1:3.0, N)
        x₂ = rand(0.0:0.1:3.0, N)
        Gₓ = FactorizedGrid(x₁, x₂)

        y₁ = rand(0.0:0.1:3.0, M)
        y₂ = rand(0.0:0.1:3.0, M)
        Gy = FactorizedGrid(y₁, y₂)

        K = kernelmatrix(k, flat_arr(Gₓ), flat_arr(Gy))

        @test kernelmatrix(k, Gₓ, Gy) ≈ K atol = 1e-8
    end
end
