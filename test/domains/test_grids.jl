using ReTest
using FunctionalGPs
using FunctionalGPs: _grid_to_colvecs
using KernelFunctions
using LinearAlgebra

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
        @test string(G) == "FactorizedGrid(($N, $M))"

        K = kernelmatrix(k, flat_arr(G))

        @test kernelmatrix(k, G) ≈ K atol = 1.0e-8
        @test kernelmatrix_diag(k, G) ≈ diag(K)
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

        @test kernelmatrix(k, Gₓ, Gy) ≈ K atol = 1.0e-8
    end

    @testset "Mixed FactorizedGrid and ColVecs" begin
        x₁ = rand(0.0:0.1:3.0, 5)
        x₂ = rand(0.0:0.1:3.0, 4)
        G = FactorizedGrid(x₁, x₂)
        G_cv = _grid_to_colvecs(G)

        # Arbitrary points as ColVecs
        pts = ColVecs(rand(2, 7))

        K_ref = kernelmatrix(k, G_cv, pts)
        @test kernelmatrix(k, G, pts) ≈ K_ref

        K_ref2 = kernelmatrix(k, pts, G_cv)
        @test kernelmatrix(k, pts, G) ≈ K_ref2
    end

    @testset "_grid_to_colvecs ordering" begin
        x₁ = [0.1, 0.2, 0.3]
        x₂ = [0.4, 0.5]
        G = FactorizedGrid(x₁, x₂)
        cv = _grid_to_colvecs(G)

        # Column-major: dimension 1 varies fastest
        # Points: (0.1,0.4), (0.2,0.4), (0.3,0.4), (0.1,0.5), (0.2,0.5), (0.3,0.5)
        expected_pts = [
            [0.1, 0.4], [0.2, 0.4], [0.3, 0.4],
            [0.1, 0.5], [0.2, 0.5], [0.3, 0.5],
        ]
        for (i, pt) in enumerate(expected_pts)
            @test cv.X[:, i] ≈ pt
        end
    end

    @testset "kernelmatrix_diag with vector-of-vectors" begin
        pts = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        pts_mat = reduce(hcat, pts)

        d_ref = kernelmatrix_diag(k, ColVecs(pts_mat))
        # Falls through to generic KernelFunctions fallback (no piracy needed)
        d_vecs = kernelmatrix_diag(k, ColVecs(pts_mat))
        @test d_vecs ≈ d_ref
    end
end
