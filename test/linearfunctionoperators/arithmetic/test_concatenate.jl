using GaussPDE
using AbstractGPs
using KernelFunctions
using ReTest

@testset "ConcatenatedLinearFunctionOperator" begin
    k₁ = WendlandKernel(1, 3)
    k₂ = WendlandKernel(1, 2)
    k = k₁ ⊗ k₂

    𝒟₁ = PartialDerivative{1, 2}(1, (2, 1))
    𝒟₂ = PartialDerivative{1, 2}(1, (0, 1))
    𝒟₃ = PartialDerivative{1, 2}(1, (1, 0))

    𝒟₁₂ = 𝒟₁ ∘ 𝒟₂

    @test 𝒟₁₂ isa AbstractConcatenatedLinearFunctionOperator
    @test 𝒟₁₂(k) == 𝒟₁(𝒟₂(k))

    𝒟₁₂₃ = 𝒟₁ ∘ 𝒟₂ ∘ 𝒟₃

    @test 𝒟₁₂₃ isa AbstractConcatenatedLinearFunctionOperator
    @test 𝒟₁₂₃(k) == 𝒟₁(𝒟₂(𝒟₃(k)))

    f = GP(WendlandKernel(1, 3))
    𝒟₁ = PartialDerivative((2,))
    𝒟₂ = PartialDerivative((1,))
    𝒟₃ = PartialDerivative((3,))
    concat = 𝒟₁ ∘ 𝒟₂
    @test concat isa AbstractConcatenatedLinearFunctionOperator{2}
    @test 𝒟₁ ∘ (𝒟₂ ∘ 𝒟₃) isa AbstractConcatenatedLinearFunctionOperator{3}
    @test (𝒟₁ ∘ 𝒟₂) ∘ 𝒟₃ isa AbstractConcatenatedLinearFunctionOperator{3}
    @test 𝒟₁ ∘ 𝒟₂ ∘ 𝒟₃ isa AbstractConcatenatedLinearFunctionOperator{3}
    @test (𝒟₁ ∘ 𝒟₂) ∘ (𝒟₃ ∘ 𝒟₃) isa AbstractConcatenatedLinearFunctionOperator{4}

    @test concat(f.mean) == 𝒟₁(𝒟₂(f.mean))
    @test concat(𝒟₃(f.kernel)) == 𝒟₁(𝒟₂(𝒟₃(f.kernel)))
    @test (concat ∘ 𝒟₃)(f.kernel) == 𝒟₁(𝒟₂(𝒟₃(f.kernel)))
    @test concat(concat(f.kernel), arg = 1) == 𝒟₁(𝒟₂(𝒟₁(𝒟₂(f.kernel)), arg = 1), arg = 1)

    δ = EvaluationFunctional(rand(3))
    δ2 = EvaluationFunctional(rand(4))
    stacked = StackedPVCrosscov([δ(f.kernel), δ2(f.kernel)])
    @test concat(stacked) ≈ 𝒟₁(𝒟₂(stacked))

    @test string(concat) == "($(string(𝒟₁))) ∘ ($(string(𝒟₂)))"
end
