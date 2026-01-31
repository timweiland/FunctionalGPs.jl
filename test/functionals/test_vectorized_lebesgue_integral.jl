using GaussPDE

@testset "VectorizedLebesgueIntegral" begin
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
