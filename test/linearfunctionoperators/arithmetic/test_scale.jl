using GaussPDE
using AbstractGPs
using KernelFunctions
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
