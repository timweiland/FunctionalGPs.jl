@testset "Linear Differential Operator" begin
    using GaussPDE

    coeffs_dict = Dict(1 => Dict((1,) => 42.0, (2,) => 1.0), 2 => Dict((3,) => 10.0))

    𝒟 = LinearDifferentialOperator{2}(coeffs_dict)
    @test 𝒟 isa LinearDifferentialOperator{3, 2, 1}
    @test 𝒟 isa AbstractSumLinearFunctionOperator{3}
    @test length(𝒟.summands) == 3
    for summand in 𝒟.summands
        @test summand isa MaybeScaledPartialDerivative{2, 1}
    end

    coeffs_dict_1D = Dict(1 => Dict((1,) => 42.0, (2,) => 1.0))
    𝒟_1D = LinearDifferentialOperator{1}(coeffs_dict_1D)
    k = WendlandKernel(1, 3)
    X = 2.0 * rand(10)
    @test kernelmatrix(𝒟_1D(k), X) ≈ kernelmatrix(
        (42.0 * PartialDerivative((1,)))(k) + PartialDerivative((2,))(k),
        X,
    )
end
