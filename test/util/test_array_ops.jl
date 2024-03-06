using ReTest
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

    @testset "reshape product broadcast" begin
        A_old = rand(4, 5)
        B_old = rand(2, 3)
        A, B = reshape_product_broadcast(A_old, B_old)
        C = A .+ B
        @test size(C) == (4, 5, 2, 3)
        for i in 1:4, j in 1:5, k in 1:2, l in 1:3
            @test C[i, j, k, l] == A_old[i, j] + B_old[k, l]
        end
    end
end