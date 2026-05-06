using FunctionalGPs

@testset "VectorizedLebesgueIntegral" begin
    @testset "Varargs construction" begin
        # Regression: the varargs form used to recurse infinitely because the
        # splatted Tuple did not match the AbstractArray inner constructor.
        i1 = Interval(0.0, 1.0)
        i2 = Interval(1.0, 2.0)
        ℒ_one = VectorizedLebesgueIntegral(i1)
        @test ℒ_one isa VectorizedLebesgueIntegral
        @test length(ℒ_one.domains) == 1
        ℒ_two = VectorizedLebesgueIntegral(i1, i2)
        @test length(ℒ_two.domains) == 2
    end

    @testset "One-sided application" begin
        w = WendlandKernel(1, 3, 7 // 10)
        domains = intervals_from_endpoints(range(0, 4; step = 0.3))
        ℒ = VectorizedLebesgueIntegral(domains)
        ℒw = ℒ(w, arg = 1)
        wℒ = ℒ(w)
        @test ℒw isa IntegralPVCrosscov
        @test wℒ isa IntegralPVCrosscov
        @test ℒw.integral_arg == 1
        @test wℒ.integral_arg == 2
    end

    @testset "Two-sided application" begin
        w = WendlandKernel(1, 3, 7 // 10)
        domains = intervals_from_endpoints(range(0, 4; step = 0.3))
        domains2 = intervals_from_endpoints(range(1, 5; step = 0.25))
        ℒ1 = VectorizedLebesgueIntegral(domains)
        ℒ2 = VectorizedLebesgueIntegral(domains2)
        K = ℒ1(ℒ2(w))
        @test K ≈ kernel_integrate_integrate(w, domains, domains2)
    end
end
