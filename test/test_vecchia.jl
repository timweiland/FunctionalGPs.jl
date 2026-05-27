using FunctionalGPs
using AbstractGPs
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Test

@testset "Vecchia: categorisation + coordinates" begin
    X = [0.0, 0.25, 0.5, 1.0]
    @test functional_category(EvaluationFunctional(X)) === EVALUATION
    @test functional_category(VectorizedLebesgueIntegral([Interval(0.0, 1.0)])) ===
        INTEGRAL

    L_dy = EvaluationFunctional(X) ∘ PartialDerivative((1,))
    @test functional_category(L_dy) === DERIVATIVE
    @test functional_category(L_dy ∘ PartialDerivative((1,))) === DERIVATIVE

    @test get_coordinates(EvaluationFunctional(X)) ==
        reshape(Float64.(X), 1, length(X))
    @test get_coordinates(L_dy) == reshape(Float64.(X), 1, length(X))

    intervals = [Interval(0.0, 1.0), Interval(1.0, 2.0)]
    @test get_coordinates(VectorizedLebesgueIntegral(intervals)) == [0.5 1.5]
end

@testset "Vecchia: bridge to GMRF" begin
    k = WendlandKernel(1, 3, 8 // 10)
    f = GP(k)
    X = collect(0.0:0.1:1.0)
    Xd = [0.25, 0.5, 0.75]
    Xq = [Interval(0.0, 0.5), Interval(0.5, 1.0)]

    fg = FunctionalGaussian(
        f;
        y = EvaluationFunctional(X),
        dy = EvaluationFunctional(Xd) ∘ PartialDerivative((1,)),
        q = VectorizedLebesgueIntegral(Xq),
    )

    g = vecchia(fg)
    @test g isa MetaGMRF
    @test length(g) == length(fg)
    @test keys(g.metadata) == (:y, :dy, :q)
    @test block_range(g, :y) == block_range(fg, :y)
    @test block_range(g, :dy) == block_range(fg, :dy)
    @test block_range(g, :q) == block_range(fg, :q)

    Q = precision_matrix(g)
    @test Q isa AbstractMatrix
    @test size(Q) == (length(fg), length(fg))
    @test issparse(Q)

    # The GMRF approximates the joint Gaussian: variance per block should be
    # in the right ballpark (within a few × of the true variance).
    Σ = Matrix(cov(fg))
    var_full = diag(Σ)
    var_gmrf = var(g)
    @test all(>(0), var_gmrf)
    # Loose check — Vecchia is an approximation; ratio shouldn't be wild.
    ratio = var_gmrf ./ var_full
    @test all(0.1 .< ratio .< 10)

    # `:natural` ordering also works.
    g_nat = vecchia(fg; ordering = :natural)
    @test g_nat isa MetaGMRF
    @test length(g_nat) == length(fg)

    # Simplicial path (λ = nothing).
    g_simp = vecchia(fg; λ = nothing)
    @test g_simp isa MetaGMRF
    @test length(g_simp) == length(fg)

    # nameview slices a flat latent into per-block NamedTuple views.
    x = randn(length(g))
    nv = nameview(g, x)
    @test nv isa NamedTuple
    @test keys(nv) == (:y, :dy, :q)
    @test nv.y == view(x, block_range(g, :y))
    @test nv.dy == view(x, block_range(g, :dy))
    @test nv.q == view(x, block_range(g, :q))
    # Writes through the view mutate the underlying latent.
    nv.y[1] = 42.0
    @test x[block_range(g, :y).start] == 42.0
    # Length mismatch errors clearly.
    @test_throws DimensionMismatch nameview(g, randn(length(g) + 1))
end
