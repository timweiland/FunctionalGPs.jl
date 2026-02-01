using FunctionalGPs
using LinearAlgebra

@testset "Factorized Integrals" begin
    k1 = WendlandKernel(1, 3, 0.8)
    k2 = WendlandKernel(1, 2, 0.5)
    k = k1 ⊗ k2

    intervals1 = intervals_from_endpoints(0:0.2:1)
    intervals2 = intervals_from_endpoints(0.15:0.1:2)
    ∫₁ = VectorizedLebesgueIntegral(intervals1)
    ∫₂ = VectorizedLebesgueIntegral(intervals2)
    ∫_combined = VectorizedLebesgueIntegral(intervals1 ⊗ intervals2)

    k∫ = ∫_combined(k)

    @testset "One-sided application" begin
        @test k∫ isa TensorProductCrosscov{2}
        @test k∫.factors[1] == ∫₁(k1)
        @test k∫.factors[2] == ∫₂(k2)
    end

    @testset "Two-sided application" begin
        ∫k∫ = ∫_combined(k∫)
        ∫₁k1∫₁ = ∫₁(∫₁(k1))
        ∫₂k2∫₂ = ∫₂(∫₂(k2))
        @test ∫k∫ ≈ kron(∫₂k2∫₂, ∫₁k1∫₁)
    end
end
