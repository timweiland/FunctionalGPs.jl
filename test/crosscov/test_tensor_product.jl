using FunctionalGPs
using FunctionalGPs: _khatri_rao_columns, _khatri_rao_rows
using KernelFunctions: KernelTensorProduct, ColVecs
using Kronecker: KroneckerProduct
using LinearAlgebra

@testset "TensorProductCrosscov" begin
    k = WendlandKernel(1, 3, 0.8)
    x₁ = 0:0.2:1
    x₂ = 0.15:0.1:2
    δ₁ = EvaluationFunctional(x₁)
    δ₂ = EvaluationFunctional(x₂)
    δ₁k = δ₁(k)
    δ₂k = δ₂(k)
    pv_prod = δ₁k ⊗ δ₂k
    @test pv_prod isa TensorProductCrosscov{2}
    @test pv_prod.factors == (δ₁k, δ₂k)

    @test_throws ArgumentError δ₁k ⊗ δ₂(k, arg = 1)

    @test string(pv_prod) == "$(string(δ₁k)) ⊗ $(string(δ₂k))"

    @testset "Factorized input" begin
        x = FactorizedGrid([0.1, 0.3], [0.15, 0.25])
        @test kernelmatrix(pv_prod, x) ≈ kron(kernelmatrix(δ₂k, x[2]), kernelmatrix(δ₁k, x[1]))
    end

    @testset "Vector-of-vectors input (one-arg, randvar_arg=2)" begin
        # Arbitrary (non-grid) 2D points
        points = [[0.1, 0.15], [0.3, 0.25], [0.5, 0.1], [0.7, 0.4]]
        K = kernelmatrix(pv_prod, points)

        # The randvar dimension (rows of k₁/k₂ factor matrices) gets Khatri-Rao'd,
        # while the points dimension stays aligned. Verify against brute-force:
        # each entry K[i, j] = ∏_d factor_d_kernelmatrix[obs_d_i, point_d_j]
        coords_1 = [p[1] for p in points]
        coords_2 = [p[2] for p in points]
        K₁ = kernelmatrix(δ₁k, coords_1)
        K₂ = kernelmatrix(δ₂k, coords_2)
        K_expected = _khatri_rao_rows([K₁, K₂])
        @test K ≈ K_expected
    end

    @testset "Vector-of-vectors input (one-arg, randvar_arg=1)" begin
        # Build a TensorProductCrosscov with randvar_arg=1
        δ₁k_arg1 = δ₁(k; arg = 1)
        δ₂k_arg1 = δ₂(k; arg = 1)
        pv_arg1 = δ₁k_arg1 ⊗ δ₂k_arg1

        points = [[0.2, 0.3], [0.4, 0.5], [0.6, 0.8]]
        K = kernelmatrix(pv_arg1, points)

        coords_1 = [p[1] for p in points]
        coords_2 = [p[2] for p in points]
        K₁ = kernelmatrix(δ₁k_arg1, coords_1)
        K₂ = kernelmatrix(δ₂k_arg1, coords_2)
        K_expected = _khatri_rao_columns([K₁, K₂])
        @test K ≈ K_expected
    end

    @testset "Two-arg kernelmatrix errors (factors lack two-arg support)" begin
        points_x = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        points_y = [[0.15, 0.25], [0.35, 0.45]]
        @test_throws MethodError kernelmatrix(pv_prod, points_x, points_y)
    end

    @testset "Vector-of-vectors matches FactorizedGrid at grid points" begin
        g1 = [0.1, 0.3, 0.5]
        g2 = [0.15, 0.25]
        grid = FactorizedGrid(g1, g2)
        K_grid = Matrix(kernelmatrix(pv_prod, grid))

        # Construct the same grid points as vector-of-vectors (column-major order)
        grid_vecs = [[x1, x2] for x2 in g2 for x1 in g1]
        K_vecs = kernelmatrix(pv_prod, grid_vecs)

        @test K_vecs ≈ K_grid
    end

    @testset "Evaluation functional" begin
        X = FactorizedGrid([0.1, 0.3], [0.15, 0.25])
        δ_factorized = EvaluationFunctional(X)
        result = δ_factorized(pv_prod)
        @test result isa KroneckerProduct
        expected = kron(kernelmatrix(δ₂k, X[2]), kernelmatrix(δ₁k, X[1]))
        @test Matrix(result) ≈ expected
    end

    k1 = WendlandKernel(1, 3, 0.8)
    k2 = WendlandKernel(1, 2, 0.4)
    pv_prod = δ₁(k1) ⊗ δ₂(k2)
    @testset "Partial Derivative" begin
        pd = PartialDerivative((1, 2))
        pd_pv_prod = pd(pv_prod)
        @test pd_pv_prod isa TensorProductCrosscov{2}
        @test pd_pv_prod.factors[1].k isa DerivativeKernel1D{1, 0}
        @test pd_pv_prod.factors[2].k isa DerivativeKernel1D{2, 0}
    end
end

@testset "EvaluationPVCrosscov with KernelTensorProduct" begin
    k1 = WendlandKernel(1, 3, 0.8)
    k2 = WendlandKernel(1, 2, 0.5)
    k_tp = k1 ⊗ k2

    # Observation points as vector-of-vectors
    obs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

    @testset "Vector-of-vectors obs, vector-of-vectors eval (arg=2)" begin
        δ = EvaluationFunctional(obs)
        pv = δ(k_tp)  # randvar_arg=2 by default

        eval_pts = [[0.15, 0.25], [0.35, 0.45], [0.55, 0.65]]
        K = kernelmatrix(pv, eval_pts)

        # Reference: direct KernelFunctions kernelmatrix with ColVecs
        obs_mat = reduce(hcat, obs)
        eval_mat = reduce(hcat, eval_pts)
        K_ref = kernelmatrix(k_tp, ColVecs(eval_mat), ColVecs(obs_mat))
        @test K ≈ K_ref
    end

    @testset "Vector-of-vectors obs, vector-of-vectors eval (arg=1)" begin
        δ = EvaluationFunctional(obs)
        pv = δ(k_tp; arg = 1)

        eval_pts = [[0.15, 0.25], [0.35, 0.45], [0.55, 0.65]]
        K = kernelmatrix(pv, eval_pts)

        obs_mat = reduce(hcat, obs)
        eval_mat = reduce(hcat, eval_pts)
        K_ref = kernelmatrix(k_tp, ColVecs(obs_mat), ColVecs(eval_mat))
        @test K ≈ K_ref
    end

    @testset "Vector-of-vectors obs, FactorizedGrid eval (arg=2)" begin
        δ = EvaluationFunctional(obs)
        pv = δ(k_tp)

        grid = FactorizedGrid([0.1, 0.3, 0.5], [0.2, 0.4])
        K = kernelmatrix(pv, grid)

        # Reference: convert both sides to ColVecs
        obs_mat = reduce(hcat, obs)
        grid_arr = convert(Array, grid)
        d = length(grid.ranges)
        grid_perm = permutedims(grid_arr, (ndims(grid_arr), 1:(ndims(grid_arr) - 1)...))
        grid_mat = reshape(grid_perm, d, prod(size(grid)))
        K_ref = kernelmatrix(k_tp, ColVecs(grid_mat), ColVecs(obs_mat))
        @test K ≈ K_ref
    end

    @testset "Vector-of-vectors obs, FactorizedGrid eval (arg=1)" begin
        δ = EvaluationFunctional(obs)
        pv = δ(k_tp; arg = 1)

        grid = FactorizedGrid([0.1, 0.3, 0.5], [0.2, 0.4])
        K = kernelmatrix(pv, grid)

        obs_mat = reduce(hcat, obs)
        grid_arr = convert(Array, grid)
        d = length(grid.ranges)
        grid_perm = permutedims(grid_arr, (ndims(grid_arr), 1:(ndims(grid_arr) - 1)...))
        grid_mat = reshape(grid_perm, d, prod(size(grid)))
        K_ref = kernelmatrix(k_tp, ColVecs(obs_mat), ColVecs(grid_mat))
        @test K ≈ K_ref
    end
end
