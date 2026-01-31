@testset "Integral-evaluation cross-covariance" begin
    k = HalfIntegerMaternKernel(2, [1.3])
    domains = [Interval(0.0, 0.4), Interval(0.4, 0.9)]
    X = collect(range(-0.2, 1.0; length = 5))

    # Test integral on first argument, evaluation on second
    pv = IntegralPVCrosscov(k, domains, 1)
    ℒ = EvaluationFunctional(X)
    lazy = ℒ(pv)
    dense = Matrix(kernel_integrate_evaluate(k, domains, X))
    @test Matrix(lazy) ≈ dense atol = 1.0e-9 rtol = 1.0e-7

    # Test integral on second argument, evaluation on first
    pv_right = IntegralPVCrosscov(k, domains, 2)
    lazy_right = ℒ(pv_right)
    dense_right = Matrix(kernel_integrate_evaluate(k, domains, X))'
    @test Matrix(lazy_right) ≈ dense_right atol = 1.0e-9 rtol = 1.0e-7
end

@testset "Integral-integral covariance" begin
    k = HalfIntegerMaternKernel(1, [0.8])
    domains_left = [Interval(-0.1, 0.2), Interval(0.2, 0.5), Interval(0.5, 0.9)]
    domains_right = [Interval(0.0, 0.3), Interval(0.3, 0.7)]

    # Test two different domains
    pv = IntegralPVCrosscov(k, domains_right, 1)
    ℒ = VectorizedLebesgueIntegral(domains_left)
    lazy = ℒ(pv)
    dense = Matrix(kernel_integrate_integrate(k, domains_left, domains_right))
    @test Matrix(lazy) ≈ dense atol = 1.0e-9 rtol = 1.0e-7

    # Test same domains (symmetric case)
    pv_same = IntegralPVCrosscov(k, domains_left, 1)
    ℒ_same = VectorizedLebesgueIntegral(domains_left)
    lazy_same = ℒ_same(pv_same)
    dense_same = Matrix(kernel_integrate_integrate(k, domains_left))
    @test Matrix(lazy_same) ≈ dense_same atol = 1.0e-9 rtol = 1.0e-7
end
