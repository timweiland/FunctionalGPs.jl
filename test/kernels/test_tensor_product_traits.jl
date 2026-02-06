using FunctionalGPs
using FunctionalGPs: KhatriRaoMatrix, StationaryKernelMatrix
using KernelFunctions: ColVecs, with_lengthscale, Matern52Kernel
using Kronecker: KroneckerProduct
using LazyArrays: BroadcastArray
using ToeplitzMatrices: SymmetricToeplitz
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

# Lazy structure tests using a kernel with StationaryKernelTrait so that
# per-dimension factor matrices are lazy (SymmetricToeplitz, StationaryKernelMatrix)
# rather than dense Matrix{Float64}.
@testset "Tensor product lazy structure (stationary kernels)" begin
    k₁ = HalfIntegerMaternKernel(2, [0.8])
    k₂ = HalfIntegerMaternKernel(2, [1.2])
    k = k₁ ⊗ k₂

    g1 = 0:0.1:1.0
    g2 = 0:0.2:2.0
    pts = [[0.1, 0.2], [0.3, 0.5], [0.7, 0.9], [0.4, 0.6]]
    pts2 = [[0.15, 0.25], [0.35, 0.55], [0.8, 1.0]]

    # Dense reference via KernelFunctions
    _to_mat(X) = reduce(hcat, X)
    _ref(X) = kernelmatrix(k, ColVecs(_to_mat(X)))
    _ref(X, Y) = kernelmatrix(k, ColVecs(_to_mat(X)), ColVecs(_to_mat(Y)))

    @testset "FactorizedGrid (one-arg) → Kronecker of SymmetricToeplitz" begin
        grid = FactorizedGrid(g1, g2)
        K = kernel_evaluate_evaluate(k, grid)
        @test K isa KroneckerProduct
        @test K.A isa SymmetricToeplitz
        @test K.B isa SymmetricToeplitz
        # correctness
        grid_vecs = [[x, y] for y in g2 for x in g1]
        @test collect(K) ≈ _ref(grid_vecs)
    end

    @testset "FactorizedGrid × FactorizedGrid (two-arg) → Kronecker of lazy" begin
        g1r = 0.05:0.1:1.05
        g2r = 0.1:0.2:2.1
        grid_l = FactorizedGrid(g1, g2)
        grid_r = FactorizedGrid(g1r, g2r)
        K = kernel_evaluate_evaluate(k, grid_l, grid_r)
        @test K isa KroneckerProduct
        @test K.A isa StationaryKernelMatrix
        @test K.B isa StationaryKernelMatrix
        # correctness
        left_vecs = [[x, y] for y in g2 for x in g1]
        right_vecs = [[x, y] for y in g2r for x in g1r]
        @test collect(K) ≈ _ref(left_vecs, right_vecs)
    end

    @testset "FactorizedGrid × vector-of-vectors → KhatriRao{1} of lazy" begin
        grid = FactorizedGrid(g1, g2)
        K = kernel_evaluate_evaluate(k, grid, pts)
        @test K isa KhatriRaoMatrix{1}
        for f in K.factors
            @test !(f isa Matrix)
        end
        # correctness
        grid_vecs = [[x, y] for y in g2 for x in g1]
        @test Matrix(K) ≈ _ref(grid_vecs, pts)
    end

    @testset "vector-of-vectors × FactorizedGrid → KhatriRao{2} of lazy" begin
        grid = FactorizedGrid(g1, g2)
        K = kernel_evaluate_evaluate(k, pts, grid)
        @test K isa KhatriRaoMatrix{2}
        for f in K.factors
            @test !(f isa Matrix)
        end
        # correctness
        grid_vecs = [[x, y] for y in g2 for x in g1]
        @test Matrix(K) ≈ _ref(pts, grid_vecs)
    end

    @testset "vector-of-vectors (one-arg) → Hadamard of lazy" begin
        K = kernel_evaluate_evaluate(k, pts)
        @test K isa BroadcastArray
        for arg in K.args
            @test !(arg isa Matrix)
        end
        @test collect(K) ≈ _ref(pts)
    end

    @testset "vector-of-vectors × vector-of-vectors → Hadamard of lazy" begin
        K = kernel_evaluate_evaluate(k, pts, pts2)
        @test K isa BroadcastArray
        for arg in K.args
            @test !(arg isa Matrix)
        end
        @test collect(K) ≈ _ref(pts, pts2)
    end
end
