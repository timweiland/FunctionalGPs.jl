using FunctionalGPs
using AbstractGPs

#=
The goal of this test:
Infer the sin function using a single direct observation and a bunch of
observations of the derivative.
=#
@testset "Sin Regression" begin
    k = WendlandKernel(1, 3, 8 // 10)
    f = GP(k)

    X = [3.14]
    y = sin.(X)

    f_1 = condition_on_observation(f, X, y, noise = 1.0e-8)
    @test f_1 isa LinearConditionalGP
    @test length(f_1.observations) == 1

    X_cos = 0:0.1:2π
    dx = PartialDerivative((1,))
    ℒ = EvaluationFunctional(X_cos) ∘ dx

    f_2 = condition_on_observation(f_1, ℒ, cos.(X_cos), noise = 1.0e-9)
    @test f_2 isa LinearConditionalGP
    @test length(f_2.observations) == 2

    X_test = 0:0.01:2π
    means, vars = mean_and_var(f_2(X_test))
    L₁_error(x, y) = max(abs.(x - y)...)
    @test L₁_error(means, sin.(X_test)) < 0.1
    @test maximum(sqrt.(vars)) < 0.1
end
