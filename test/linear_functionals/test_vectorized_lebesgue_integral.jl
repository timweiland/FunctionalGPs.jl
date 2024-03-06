using GaussPDE

@testset "VectorizedLebesgueIntegral" begin
    @testset "One-sided application" begin
        w = WendlandKernel(1, 3, 7 // 10)
        domains = intervals_from_endpoints(range(0, 4; step = 0.3))
        ℒ = VectorizedLebesgueIntegral(domains)
        ℒw = ℒ(w, arg=1)
        wℒ = ℒ(w)
        @test ℒw isa CompactPolynomialCovFunc1D_Identity_LebesgueIntegral
        @test wℒ isa CompactPolynomialCovFunc1D_Identity_LebesgueIntegral
        @test ℒw.randvar_arg == 1
        @test wℒ.randvar_arg == 2
    end

    @testset "Two-sided application" begin
        w = WendlandKernel(1, 3, 7 // 10)
        domains = intervals_from_endpoints(range(0, 4; step = 0.3))
        domains2 = intervals_from_endpoints(range(1, 5; step = 0.25))
        ℒ1 = VectorizedLebesgueIntegral(domains)
        ℒ2 = VectorizedLebesgueIntegral(domains2)
        K = ℒ1(ℒ2(w))
        @test K ≈ GaussPDE.integrate(w, domains, domains2)
    end
end