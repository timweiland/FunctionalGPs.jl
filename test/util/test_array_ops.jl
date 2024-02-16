using Test
using GaussPDE
import Random: seed!

@testset "Array ops" begin
    seed!(89345023)

    @testset "moveaxis" begin
        A = rand(3, 4, 5)
        @test moveaxis(A, 1, 1) == A
        @test moveaxis(A, 1, 2) == permutedims(A, (2, 1, 3))
        @test moveaxis(A, 1, 3) == permutedims(A, (2, 3, 1))
        @test moveaxis(A, 2, 1) == permutedims(A, (2, 1, 3))
        @test moveaxis(A, 2, 2) == A
        @test moveaxis(A, 2, 3) == permutedims(A, (1, 3, 2))
        @test moveaxis(A, 3, 1) == permutedims(A, (3, 1, 2))
        @test moveaxis(A, 3, 2) == permutedims(A, (1, 3, 2))
        @test moveaxis(A, 3, 3) == A
    end
end