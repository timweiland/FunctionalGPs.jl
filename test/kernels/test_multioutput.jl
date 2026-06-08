using FunctionalGPs
using AbstractGPs
using KernelFunctions
using KernelFunctions: with_lengthscale, SqExponentialKernel
using LinearAlgebra
using Random

@testset "Multi-output via Select" begin
    k1 = HalfIntegerMaternKernel(2, 0.3)
    k2 = with_lengthscale(SqExponentialKernel(), 0.5)   # heterogeneous: different kernel per output
    kmo = BlockDiagonalKernel(k1, k2)
    f = GP(kmo)

    X1 = collect(0.0:0.2:1.0)
    X2 = collect(0.1:0.2:0.9)

    ℒ1 = EvaluationFunctional(X1) ∘ Select(1)
    ℒ2 = EvaluationFunctional(X2) ∘ Select(2)

    @testset "diagonal blocks == single-output covariance" begin
        fg = FunctionalGaussian(f; a = ℒ1, b = ℒ2)
        δ1, δ2 = EvaluationFunctional(X1), EvaluationFunctional(X2)
        @test Matrix(cov(fg, :a)) ≈ Matrix(δ1(δ1(k1)))
        @test Matrix(cov(fg, :b)) ≈ Matrix(δ2(δ2(k2)))
    end

    @testset "cross blocks vanish (independence short-circuit)" begin
        fg = FunctionalGaussian(f; a = ℒ1, b = ℒ2)
        C = Matrix(cov(fg, :a, :b))
        @test size(C) == (length(X1), length(X2))
        @test all(iszero, C)
    end

    @testset "joint covariance is symmetric PSD" begin
        fg = FunctionalGaussian(f; a = ℒ1, b = ℒ2)
        Σ = Matrix(cov(fg))
        @test Σ ≈ Σ'
        @test isposdef(Symmetric(Σ) + 1.0e-9I)
    end

    @testset "derivative of an output" begin
        ℒ1d = EvaluationFunctional(X1) ∘ PartialDerivative((1,)) ∘ Select(1)
        fg = FunctionalGaussian(f; a = ℒ1, da = ℒ1d)
        ref = EvaluationFunctional(X1) ∘ PartialDerivative((1,))
        # diagonal derivative block matches the single-output derivative covariance
        @test Matrix(cov(fg, :da)) ≈ Matrix(ref(ref(k1)))
        # same output → value/derivative are correlated (nonzero cross-block)
        @test !all(iszero, Matrix(cov(fg, :a, :da)))
    end

    @testset "Select on a non-MO kernel errors" begin
        g = GP(k1)
        @test_throws Exception FunctionalGaussian(g; a = EvaluationFunctional(X1) ∘ Select(1))
    end

    @testset "posterior / rand reuse FunctionalGaussian" begin
        fg = FunctionalGaussian(f; a = ℒ1, b = ℒ2)
        y1 = sin.(2π .* X1)

        post = posterior(fg, (; a = y1); noise = (; a = 1.0e-6))
        @test keys(post) == (:b,)
        # independence: conditioning output 1 leaves output 2 at its prior
        @test mean(post.b) ≈ mean(fg.b)
        @test Matrix(cov(post.b)) ≈ Matrix(cov(fg.b))

        s = rand(MersenneTwister(0), fg)
        @test keys(s) == (:a, :b)
        @test length(s.a) == length(X1)
        @test length(s.b) == length(X2)
    end
end
