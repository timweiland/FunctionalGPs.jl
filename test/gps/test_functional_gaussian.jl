using FunctionalGPs
using AbstractGPs
using KernelFunctions: with_lengthscale, SqExponentialKernel
using LinearAlgebra
using Distributions
using ForwardDiff
using Random
using Test

@testset "FunctionalGaussian" begin
    k = WendlandKernel(1, 3, 8 // 10)
    f = GP(k)

    X = [0.0, 0.5, 1.0]
    Xd = [0.25, 0.75]
    L_y = EvaluationFunctional(X)
    L_dy = EvaluationFunctional(Xd) ∘ PartialDerivative((1,))
    L_q = VectorizedLebesgueIntegral([Interval(0.0, 1.0)])

    fg_yd() = FunctionalGaussian(f; y = L_y, dy = L_dy)
    fg_ydq() = FunctionalGaussian(f; y = L_y, dy = L_dy, q = L_q)

    @testset "Construction and Accessors" begin
        fg = fg_yd()
        @test keys(fg) == (:y, :dy)
        @test length(fg) == length(X) + length(Xd)
        @test block_range(fg, :y) == 1:length(X)
        @test block_range(fg, :dy) == (length(X) + 1):(length(X) + length(Xd))

        @test size(mean(fg)) == (length(fg),)
        @test size(cov(fg)) == (length(fg), length(fg))
        @test size(mean(fg, :y)) == (length(X),)
        @test size(cov(fg, :y)) == (length(X), length(X))
        @test size(cov(fg, :y, :dy)) == (length(X), length(Xd))
        @test Matrix(cov(fg, :y)) ≈ L_y(L_y(f.kernel))

        mvn = as_mvn(fg)
        @test length(mvn) == length(fg)
        @test mvn isa MvNormal

        fg_nt = FunctionalGaussian(f, (; y = L_y, dy = L_dy))
        @test keys(fg_nt) == keys(fg)
        @test mean(fg_nt) ≈ mean(fg)
        @test Matrix(cov(fg_nt)) ≈ Matrix(cov(fg))
    end

    @testset "Marginalisation matches direct application" begin
        fg = fg_ydq()

        for (name, ℒ) in ((:y, L_y), (:dy, L_dy), (:q, L_q))
            mvn_marg = marginal(fg, name)
            mvn_direct = ℒ(f)
            @test mean(mvn_marg) ≈ mean(mvn_direct)
            @test cov(mvn_marg) ≈ cov(mvn_direct)
        end

        # Multi-block marginal: order respected.
        mvn_yq = marginal(fg, (:y, :q))
        @test length(mvn_yq) == length(X) + 1
        @test mean(mvn_yq)[1:length(X)] ≈ mean(L_y(f))
        @test mean(mvn_yq)[end:end] ≈ mean(L_q(f))
    end

    @testset "posterior matches GP-posterior route" begin
        fg = fg_ydq()
        y_obs = [0.1, 0.5, 0.9]
        σ² = 0.01

        post = posterior(fg, (; y = y_obs); noise = (; y = σ²))
        @test keys(post) == (:dy, :q)

        f_post = condition_on_observation(f, L_y, y_obs; noise = σ²)
        for (name, ℒ) in ((:dy, L_dy), (:q, L_q))
            mvn_ref = ℒ(f_post)
            @test mean(post[name]) ≈ mean(mvn_ref)
            @test cov(post[name]) ≈ cov(mvn_ref)
        end
    end

    @testset "posterior on multiple observed blocks" begin
        fg = fg_ydq()
        y_obs = [0.1, 0.5, 0.9]
        q_obs = [0.4]
        σ² = 0.01

        post = posterior(fg, (; y = y_obs, q = q_obs); noise = (; y = σ², q = 1.0e-8))
        @test keys(post) == (:dy,)

        f_post = condition_on_observation(f, L_y, y_obs; noise = σ²)
        f_post = condition_on_observation(f_post, L_q, q_obs; noise = 1.0e-8)
        mvn_ref = L_dy(f_post)
        @test mean(post.dy) ≈ mean(mvn_ref)
        @test cov(post.dy) ≈ cov(mvn_ref)
    end

    @testset "Marginal log-likelihood" begin
        fg = fg_yd()
        y_obs = [0.1, 0.5, 0.9]

        σ² = 0.01
        ll = loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))
        @test ll ≈ logpdf(L_y(f; noise = σ²), y_obs)

        v = [0.01, 0.02, 0.03]
        ll_v = loglikelihood(fg, (; y = y_obs); noise = (; y = v))
        @test ll_v ≈ logpdf(L_y(f; noise = Diagonal(v)), y_obs)

        M = Matrix(0.01 * I(3))
        ll_M = loglikelihood(fg, (; y = y_obs); noise = (; y = M))
        @test ll_M ≈ logpdf(L_y(f; noise = M), y_obs)
    end

    @testset "Property access" begin
        fg = fg_yd()

        @test fg.y isa LazyMvNormal
        @test mean(fg.y) ≈ mean(L_y(f))
        @test cov(fg.y) ≈ cov(L_y(f))

        @test propertynames(fg) == (:y, :dy)
        @test :μ in propertynames(fg, true)
        @test :Σ in propertynames(fg, true)

        @test getfield(fg, :μ) === mean(fg)
        @test getfield(fg, :Σ) === cov(fg)

        # Block-name access wins when a block shadows an internal field.
        fg_shadow = FunctionalGaussian(f; μ = L_y)
        @test fg_shadow.μ isa LazyMvNormal
        @test getfield(fg_shadow, :μ) isa AbstractVector
    end

    @testset "LazyMvNormal preserves block structure" begin
        # Stationary kernel produces structured per-block matrices that must
        # flow through fg.<name> unchanged for fast downstream matvec.
        k_se = with_lengthscale(SqExponentialKernel(), 0.3)
        f_se = GP(k_se)
        L_a = EvaluationFunctional(collect(0.0:0.1:1.0))
        L_b = EvaluationFunctional(collect(0.5:0.05:0.95)) ∘ PartialDerivative((1,))
        fg_struct = FunctionalGaussian(f_se; a = L_a, b = L_b)

        @test cov(fg_struct, :a) isa FunctionalGPs.CovarianceMatrix
        @test cov(fg_struct, :a, :b) isa FunctionalGPs.CovarianceMatrix
        @test cov(fg_struct, :b, :a) isa FunctionalGPs.CovarianceMatrix

        @test cov(fg_struct.a) === cov(fg_struct, :a)

        x = randn(length(fg_struct.a))
        @test isfinite(logpdf(fg_struct.a, x))

        mvn = MvNormal(fg_struct.a)
        @test mvn isa MvNormal
        @test cov(mvn) ≈ Matrix(cov(fg_struct, :a))
    end

    @testset "rand returns NamedTuple of samples" begin
        fg = fg_ydq()
        rng = MersenneTwister(42)

        s = rand(rng, fg)
        @test s isa NamedTuple
        @test keys(s) == keys(fg)
        @test s.y isa AbstractVector && length(s.y) == length(X)
        @test s.dy isa AbstractVector && length(s.dy) == length(Xd)
        @test s.q isa AbstractVector && length(s.q) == 1

        # Concatenating the block samples equals a draw from `as_mvn(fg)` under
        # the same RNG — the split is index-preserving, not a re-draw.
        rng2 = MersenneTwister(42)
        x_flat = rand(rng2, as_mvn(fg))
        @test vcat(s.y, s.dy, s.q) ≈ x_flat

        # Multi-sample form returns matrices with n columns per block.
        n = 4
        S = rand(MersenneTwister(7), fg, n)
        @test keys(S) == keys(fg)
        @test size(S.y) == (length(X), n)
        @test size(S.dy) == (length(Xd), n)
        @test size(S.q) == (1, n)

        # No-RNG forms exist and return the right shape.
        s0 = rand(fg)
        @test keys(s0) == keys(fg)
        S0 = rand(fg, 3)
        @test size(S0.y) == (length(X), 3)
    end

    @testset "_block_matmul dimension check" begin
        fg = fg_yd()
        @test_throws DimensionMismatch FunctionalGPs._block_matmul(
            fg, (1, 2), (1,), randn(length(X) + 1),
        )
    end

    @testset "Noise eltype propagates" begin
        # ForwardDiff Duals / BigFloat hyperparameters flow through only if the
        # noise covariance assembly drops Float64 hard-coding.
        fg = fg_yd()
        y_obs = [0.1, 0.5, 0.9]

        ll_big = loglikelihood(fg, (; y = y_obs); noise = (; y = BigFloat("0.01")))
        @test ll_big isa BigFloat

        v_big = BigFloat[0.01, 0.02, 0.03]
        ll_v = loglikelihood(fg, (; y = y_obs); noise = (; y = v_big))
        @test ll_v isa BigFloat
    end

    @testset "ForwardDiff through loglikelihood" begin
        # Exercise the actual Turing inference shape: kernel hyperparameter and
        # noise variance both come in as Duals, kernel + GP + FunctionalGaussian
        # are rebuilt from scratch, and the gradient must trace through block
        # assembly, noise assembly, symmetrisation, Cholesky, and logpdf.
        X_train = collect(0.0:0.2:1.0)
        y_train = sin.(2π .* X_train)

        function nll(θ)
            logell, logσ² = θ
            kk = with_lengthscale(SqExponentialKernel(), exp(logell))
            ff = GP(kk)
            fg_h = FunctionalGaussian(ff; y = EvaluationFunctional(X_train))
            return -loglikelihood(fg_h, (; y = y_train); noise = (; y = exp(logσ²)))
        end

        θ0 = [log(0.5), log(0.1)]
        @test isfinite(nll(θ0))
        g = ForwardDiff.gradient(nll, θ0)
        @test all(isfinite, g)
        @test !iszero(g)
    end

    @testset "Error handling" begin
        fg = fg_yd()

        @test_throws ArgumentError FunctionalGaussian(f, NamedTuple())
        @test_throws ArgumentError block_range(fg, :nope)
        @test_throws ArgumentError marginal(fg, :nope)
        @test_throws ArgumentError posterior(fg, (; nope = [0.0]))
        @test_throws ArgumentError posterior(
            fg, (; y = [0.1, 0.5, 0.9]); noise = (; dy = 0.01),
        )
        @test_throws ArgumentError posterior(
            fg, (; y = [0.1, 0.5, 0.9], dy = [0.0, 0.0]),
        )
        @test_throws DimensionMismatch posterior(
            fg, (; y = [0.1, 0.5, 0.9]); noise = (; y = [0.01, 0.02]),
        )
    end
end
