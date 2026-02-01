using FunctionalGPs

@testset "Interval" begin
    # Test Interval constructor
    @test_throws ArgumentError Interval(5, 3)

    # Test ndims function
    @test Base.ndims(Interval(3, 5)) == 1

    # Test getindex function
    @test Base.getindex(Interval(3, 5), 1) == 3
    @test Base.getindex(Interval(3, 5), 2) == 5
    @test_throws ArgumentError Base.getindex(Interval(3, 5), 3)

    # Test volume function
    @test volume(Interval(3, 5)) == 2

    # Test in function
    @test in(4, Interval(3, 5))
    @test !in(2, Interval(3, 5))
    @test !in(6, Interval(3, 5))

    # Test uniform_grid_n function
    @test uniform_grid_n(Interval(3, 5), 3) == [3.0, 4.0, 5.0]

    # Test uniform_grid_step function
    @test uniform_grid_step(Interval(3, 5), 0.5) == [3.0, 3.5, 4.0, 4.5, 5.0]

    # Test intervals_from_endpoints function
    @test intervals_from_endpoints([3, 5, 7]) == [Interval(3, 5), Interval(5, 7)]
end
