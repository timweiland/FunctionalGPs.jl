using FunctionalGPs
using AbstractGPs
using KernelFunctions
using LinearAlgebra
import KernelFunctions: ScaledKernel

function Base.isapprox(k1::ScaledKernel, k2::ScaledKernel)
    return k1.σ² ≈ k2.σ² && k1.kernel ≈ k2.kernel
end

@testset "ScaledLinearFunctionOperator" begin
    𝒟₁ = PartialDerivative((1,))
    𝒟₂ = PartialDerivative((2,))

    @test 42.0 * 𝒟₁ isa ConstantScaledLinearFunctionOperator{PartialDerivative{1, 1}}
    @test 42 * 𝒟₁ isa ConstantScaledLinearFunctionOperator{PartialDerivative{1, 1}}
    @test 10 // 8 * 𝒟₁ isa ConstantScaledLinearFunctionOperator{PartialDerivative{1, 1}}
    @test 1 * 𝒟₁ isa PartialDerivative

    f = GP(WendlandKernel(1, 3))
    scaled_funcop = 42.0 * 𝒟₁
    @test scaled_funcop(f.mean) isa ZeroMean
    @test scaled_funcop(𝒟₂(f.kernel), arg = 1) ≈ 42.0 * 𝒟₁(𝒟₂(f.kernel), arg = 1)
    @test scaled_funcop(scaled_funcop(f.kernel)) ≈
        42.0^2 * 𝒟₁(𝒟₁(f.kernel))

    δ1 = EvaluationFunctional(rand(10))
    δ2 = EvaluationFunctional(rand(4))
    stacked = StackedPVCrosscov([δ1(f.kernel), δ2(f.kernel)])
    scaled_stacked = scaled_funcop(stacked)
    @test scaled_stacked ≈ 42.0 * 𝒟₁(stacked)

    @test string(scaled_funcop) == "42.0 * ($(string(𝒟₁)))"
end

@testset "Scaled operator squares into self-covariance" begin
    # Regression: a scaled operator c·𝒟 applied to *both* kernel arguments must
    # contribute c² to the resulting covariance. A negative scale (e.g. -∂²)
    # used to drop one factor when both applications were scaled, because the
    # ConstantScaledPVCrosscov fold re-applied the operator scalar — yielding a
    # negative-definite "covariance" matrix.
    k = HalfIntegerMaternKernel(2, 1.0)
    X = collect(0.0:0.2:1.0)
    ∂² = PartialDerivative((2,))

    L_pos = EvaluationFunctional(X) ∘ ∂²
    L_neg = EvaluationFunctional(X) ∘ (-∂²)

    M_pos = Matrix(L_pos(L_pos(k)))
    M_neg = Matrix(L_neg(L_neg(k)))

    # (-1)² = 1: negative second derivative gives the same covariance.
    @test M_neg ≈ M_pos
    # ...and it is a genuine covariance: positive variances, positive definite.
    @test all(>(0), diag(M_pos))
    @test isposdef(Symmetric(M_neg))

    # A general scale c enters squared in the self-covariance.
    c = 2.5
    L_c = EvaluationFunctional(X) ∘ (c * ∂²)
    @test Matrix(L_c(L_c(k))) ≈ c^2 .* M_pos
end
