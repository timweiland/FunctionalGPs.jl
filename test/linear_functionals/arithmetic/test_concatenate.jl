using GaussPDE
using AbstractGPs
using KernelFunctions
import KernelFunctions: kernelmatrix

@testset "LinFctlLinFuncOpConcat" begin
    δ = EvaluationFunctional(rand(10))
    δ2 = EvaluationFunctional(rand(4))
    𝒟 = PartialDerivative((2,))
    𝒟₂ = PartialDerivative((1,))
    𝒟₃ = PartialDerivative((3,))

    f = GP(WendlandKernel(1, 3))
    concat = δ ∘ 𝒟

    @test concat isa AbstractLinFctlLinFuncOpConcat{1}
    @test δ ∘ (𝒟 ∘ 𝒟₂) isa AbstractLinFctlLinFuncOpConcat{2}
    @test (δ ∘ 𝒟) ∘ 𝒟₂ isa AbstractLinFctlLinFuncOpConcat{2}
    @test δ ∘ 𝒟 ∘ 𝒟₂ isa AbstractLinFctlLinFuncOpConcat{2}
    @test (δ ∘ 𝒟) ∘ (𝒟₂ ∘ 𝒟₃) isa AbstractLinFctlLinFuncOpConcat{3}

    @test concat(f.mean) == δ(𝒟(f.mean))
    @test concat(δ2(f.kernel)) ≈ δ(𝒟(δ2(f.kernel)))
    @test concat(concat(f.kernel)) ≈ δ(𝒟(δ(𝒟(f.kernel))))

    stacked = StackedPVCrosscov([δ(f.kernel), δ2(f.kernel)])
    @test concat(stacked) ≈ δ(𝒟(stacked))

    C = 5 * rand()
    @test concat(C * δ(f.kernel)) ≈ C * concat(δ(f.kernel))

    concat_f = concat(f)
    @test mean(concat_f) ≈ δ(𝒟(f.mean))
    @test cov(concat_f) ≈ δ(𝒟(δ(𝒟(f.kernel))))

    @test string(concat) == "$(string(δ)) ∘ ($(string(𝒟)))"
end
