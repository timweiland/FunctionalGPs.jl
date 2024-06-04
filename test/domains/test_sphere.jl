using GaussPDE

@testset "Sphere" begin
    # Test Sphere constructor
    @test_throws ArgumentError Sphere(-1)

    # Test ndims function
    @test Base.ndims(Sphere(3)) == 3

    # Test getindex function
    @test Base.getindex(Sphere(3), 1) == 3
    @test_throws ArgumentError Base.getindex(Sphere(3), 2)

    # Test volume function
    @test volume(Sphere(3)) ≈ 113.09733552923255

    # Test in function
    @test in([0, 0, 3], Sphere(3))
    @test !in([0, 0, 2], Sphere(3))
    @test !in([0, 0, 4], Sphere(3))

    # Test uniform_grid_n function
    # TODO

    # Test uniform_grid_step function
    # TODO