using GaussPDE

@testset "Box" begin
    @testset "Box constructor" begin
        # Test invalid bounds
        @test_throws ArgumentError Box((1, 0))
        @test_throws ArgumentError Box((0, 1), (1, 0))
    end

    @testset "volume" begin
        box1 = Box((0, 1), (0, 1))
        @test volume(box1) == 1.0

        box2 = Box((-1, 1), (-1, 1), (-1, 1))
        @test volume(box2) == 8.0
    end

    @testset "in" begin
        box = Box((0, 1), (0, 1))

        @test in([0.5, 0.5], box) == true
        @test in([1.5, 0.5], box) == false
        @test in([0.5, -0.5], box) == false
    end

    @testset "uniform_grid_n" begin
        box = Box((0, 1), (0, 1))

        grid = uniform_grid_n(box, 2, 2)
        @test grid isa FactorizedGrid
        @test length(grid) == 4
        @test grid[2, 1] == [1, 0]

        grid = uniform_grid_n(box, 3, 3)
        @test grid isa FactorizedGrid
        @test length(grid) == 9
        @test grid[2, 3] == [0.5, 1]
    end

    @testset "uniform_grid_step" begin
        box = Box((0, 1), (0, 1))

        grid = uniform_grid_step(box, 0.5, 0.5)
        @test length(grid) == 9
        @test grid[2, 1] == [0.5, 0]

        grid = uniform_grid_step(box, 0.25, 0.25)
        @test length(grid) == 25
        @test grid[3, 2] == [0.5, 0.25]
    end
end