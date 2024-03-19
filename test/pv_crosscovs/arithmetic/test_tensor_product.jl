using GaussPDE
using LinearAlgebra

@testset "TensorProductCrosscov" begin
    k = WendlandKernel(1, 3, 0.8)
    x₁ = 0:0.2:1
    x₂ = 0.15:0.1:2
    δ₁ = EvaluationFunctional(x₁)
    δ₂ = EvaluationFunctional(x₂)
    δ₁k = δ₁(k)
    δ₂k = δ₂(k)
    pv_prod = δ₁k ⊗ δ₂k
    @test pv_prod isa TensorProductCrosscov{2}
    @test pv_prod.factors == (δ₁k, δ₂k)

    @test_throws ArgumentError δ₁k ⊗ δ₂(k, arg=1)

    @test string(pv_prod) == "$(string(δ₁k)) ⊗ $(string(δ₂k))"

    # TODO: Uncomment me when generic kernelmatrix is implemented
    # @testset "Vector input" begin
    #     x = [0.1, 0.3]
    #     @test kernelmatrix(pv_prod, x) ≈ kernelmatrix(δ₁k, x) .* kernelmatrix(δ₂k, x)
    # end

    @testset "Factorized input" begin
        x = FactorizedGrid([0.1, 0.3], [0.15, 0.25])
        @test kernelmatrix(pv_prod, x) ≈ kron(kernelmatrix(δ₂k, x[2]), kernelmatrix(δ₁k, x[1]))
    end

    k1 = WendlandKernel(1, 3, 0.8)
    k2 = WendlandKernel(1, 2, 0.4)
    pv_prod = δ₁(k1) ⊗ δ₂(k2)
    @testset "Partial Derivative" begin
        pd = PartialDerivative((1, 2))
        pd_pv_prod = pd(pv_prod)
        @test pd_pv_prod isa TensorProductCrosscov{2}
        @test pd_pv_prod.factors[1].k isa DerivativeKernel1D{1, 0}
        @test pd_pv_prod.factors[2].k isa DerivativeKernel1D{2, 0}
    end
end