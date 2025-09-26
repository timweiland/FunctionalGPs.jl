using GaussPDE
using AbstractGPs
using KernelFunctions
import KernelFunctions: kernelmatrix, KernelSum

function Base.isapprox(k1::KernelSum, k2::KernelSum)
    return all(k1.kernels .≈ k2.kernels)
end

@testset "SumLinearFunctionOperator" begin
    𝒟₁ = PartialDerivative((1,))
    𝒟₂ = PartialDerivative((2,))
    𝒟₃ = PartialDerivative((3,))
    𝒟₄ = PartialDerivative((4,))

    @test 𝒟₁ + 𝒟₂ isa AbstractSumLinearFunctionOperator{2}
    @test 𝒟₁ + (𝒟₂ + 𝒟₃) isa AbstractSumLinearFunctionOperator{3}
    @test (𝒟₁ + 𝒟₂) + 𝒟₃ isa AbstractSumLinearFunctionOperator{3}
    @test 𝒟₁ + 𝒟₂ + 𝒟₃ isa AbstractSumLinearFunctionOperator{3}
    @test (𝒟₁ + 𝒟₂) + (𝒟₃ + 𝒟₄) isa AbstractSumLinearFunctionOperator{4}

    f = GP(WendlandKernel(1, 3))
    sum_funcop = 𝒟₁ + 𝒟₂
    @test sum_funcop(f.mean) isa ZeroMean
    @test sum_funcop(𝒟₃(f.kernel), arg = 1) ≈ 𝒟₁(𝒟₃(f.kernel), arg = 1) + 𝒟₂(𝒟₃(f.kernel), arg = 1)
    @test sum_funcop(sum_funcop(f.kernel)) ≈
        𝒟₁(𝒟₁(f.kernel)) + 𝒟₁(𝒟₂(f.kernel)) + 𝒟₂(𝒟₁(f.kernel)) + 𝒟₂(𝒟₂(f.kernel))

    δ1 = EvaluationFunctional(rand(10))
    δ2 = EvaluationFunctional(rand(4))
    stacked = StackedPVCrosscov([δ1(f.kernel), δ2(f.kernel)])
    sum_stacked = sum_funcop(stacked)
    @test sum_stacked ≈ 𝒟₁(stacked) + 𝒟₂(stacked)

    @test string(sum_funcop) == "($(string(𝒟₁))) + ($(string(𝒟₂)))"
end
