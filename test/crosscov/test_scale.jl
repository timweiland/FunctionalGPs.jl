using FunctionalGPs
using KernelFunctions

@testset "ScaledPVCrosscov" begin
    k = WendlandKernel(1, 3)
    δ = EvaluationFunctional(rand(5))
    kδ = δ(k)
    scaled = 42.0 * kδ
    @test scaled isa AbstractScaledPVCrosscov
    @test scaled isa ConstantScaledPVCrosscov
    @test 1 * scaled == scaled
    @test string(scaled) == "42.0 * ($(string(kδ)))"
    X = rand(5)
    @test kernelmatrix(scaled, X) ≈ 42.0 * kernelmatrix(kδ, X)
    @test 5 * scaled ≈ (5 * 42.0) * kδ
end
