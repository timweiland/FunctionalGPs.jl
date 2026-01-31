using GaussPDE
using AbstractGPs
using KernelFunctions
using Distributions
import KernelFunctions: kernelmatrix

@testset "SumLinearFunctional" begin
    δ1 = EvaluationFunctional(rand(10))
    δ2 = EvaluationFunctional(rand(10))
    δ3 = EvaluationFunctional(rand(10))
    δ4 = EvaluationFunctional(rand(10))
    δ_err = EvaluationFunctional(rand(4))
    @test δ1 + δ2 isa AbstractSumLinearFunctional{2}
    @test δ1 + (δ2 + δ3) isa AbstractSumLinearFunctional{3}
    @test (δ1 + δ2) + δ3 isa AbstractSumLinearFunctional{3}
    @test δ1 + δ2 + δ3 isa AbstractSumLinearFunctional{3}
    @test (δ1 + δ2) + (δ3 + δ4) isa AbstractSumLinearFunctional{4}
    @test_throws ArgumentError δ1 + δ_err

    f = GP(WendlandKernel(1, 3))
    sum_fctl = δ1 + δ2
    @test sum_fctl(f.mean) == δ1(f.mean) + δ2(f.mean)
    @test sum_fctl(δ3(f.kernel)) ≈ δ1(δ3(f.kernel)) + δ2(δ3(f.kernel))
    @test sum_fctl(sum_fctl(f.kernel)) ≈
        δ1(δ1(f.kernel)) + δ1(δ2(f.kernel)) + δ2(δ1(f.kernel)) + δ2(δ2(f.kernel))

    stacked = StackedPVCrosscov([δ1(f.kernel), δ2(f.kernel)])
    sum_stacked = sum_fctl(stacked)
    @test sum_stacked ≈ δ1(stacked) + δ2(stacked)

    C = 5 * rand()
    @test sum_fctl(C * δ3(f.kernel)) ≈ C * sum_fctl(δ3(f.kernel))

    sum_f = sum_fctl(f)
    @test Distributions.mean(sum_f) ≈ δ1(f.mean) + δ2(f.mean)
    @test cov(sum_f) ≈
        δ1(δ1(f.kernel)) + δ1(δ2(f.kernel)) + δ2(δ1(f.kernel)) + δ2(δ2(f.kernel))

    @test string(sum_fctl) == "($(string(δ1))) + ($(string(δ2)))"
end
