using GaussPDE

@testset "FactorizedBox" begin
    interval_vec1 = [Interval(0, 1), Interval(2, 3)]
    interval_vec2 = [Interval(0, 1), Interval(2, 3), Interval(4, 5)]
    domains = FactorizedBoxDomains([interval_vec1, interval_vec2])

    @test ndims(domains) == 2
    @test length(domains) == 6
    @test size(domains) == (2, 3)

    @test getindex(domains, CartesianIndex(2, 3)) == BoxDomain(Interval(2, 3), Interval(4, 5))
    @test getindex(domains, 2, 3) == BoxDomain(Interval(2, 3), Interval(4, 5))
    @test getindex(domains, 6) == BoxDomain(Interval(2, 3), Interval(4, 5))

    @test get_intervals(domains, 1) == interval_vec1
    @test get_intervals(domains) == [interval_vec1, interval_vec2]

    @testset "⊗ operator" begin
        domains = interval_vec1 ⊗ interval_vec2
        @test ndims(domains) == 2
        @test length(domains) == 6
        @test size(domains) == (2, 3)
        @test getindex(domains, CartesianIndex(2, 3)) == BoxDomain(Interval(2, 3), Interval(4, 5))
    end
end