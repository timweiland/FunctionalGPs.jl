using GaussPDE
using KernelFunctions
using ReTest

@testset "ConcatenatedLinearFunctionOperator" begin
    k₁ = WendlandKernel(1, 3)
    k₂ = WendlandKernel(1, 2)
    k = k₁ ⊗ k₂

    𝒟₁ = PartialDerivative{1,2}(1, (2, 1))
    𝒟₂ = PartialDerivative{1,2}(1, (0, 1))
    𝒟₃ = PartialDerivative{1,2}(1, (1, 0))

    𝒟₁₂ = 𝒟₁ ∘ 𝒟₂

    @test 𝒟₁₂ isa AbstractConcatenatedLinearFunctionOperator
    @test 𝒟₁₂(k) == 𝒟₁(𝒟₂(k))

    𝒟₁₂₃ = 𝒟₁ ∘ 𝒟₂ ∘ 𝒟₃

    @test 𝒟₁₂₃ isa AbstractConcatenatedLinearFunctionOperator
    @test 𝒟₁₂₃(k) == 𝒟₁(𝒟₂(𝒟₃(k)))
end