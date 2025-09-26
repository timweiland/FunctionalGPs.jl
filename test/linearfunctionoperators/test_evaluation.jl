using GaussPDE
using ReTest
using Random: seed!

@testset "EvaluationFunctional" begin
    seed!(237840)

    k = WendlandKernel(1, 3)
    X₁ = RowVecs(rand(7))
    X₂ = RowVecs(rand(8))

    δ₁ = EvaluationFunctional(X₁)
    δ₂ = EvaluationFunctional(X₂)

    @test δ₁(k) isa EvaluationPVCrosscov
    @test δ₂(k) isa EvaluationPVCrosscov

    K₁₂ = δ₁(δ₂(k))
    K₂₁ = δ₂(δ₁(k))
    K₁₁ = δ₁(δ₁(k))
    K₂₂ = δ₂(δ₂(k))

    @test kernelmatrix(k, X₁, X₂) ≈ K₁₂
    @test kernelmatrix(k, X₂, X₁) ≈ K₂₁
    @test kernelmatrix(k, X₁, X₁) ≈ K₁₁
    @test kernelmatrix(k, X₂, X₂) ≈ K₂₂
end
