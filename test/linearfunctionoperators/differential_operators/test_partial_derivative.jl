@testset "Partial Derivative" begin
    using GaussPDE
    using ReTest

    @testset "Creation" begin
        pd1 = PartialDerivative{2, 3}(2, (1, 2, 3))
        @test pd1.output_idx == 2
        @test pd1.multi_idx == (1, 2, 3)

        @test_throws DomainError PartialDerivative{2, 1}(3, (1,))

        pd2 = PartialDerivative((1, 2, 3))
        @test pd2 isa PartialDerivative{1, 3}
        @test pd2.output_idx == 1

        pd3 = PartialDerivative{2}(1, (1, 2, 3))
        @test pd3 isa PartialDerivative{2, 3}
    end

    @testset "Show" begin
        pd = PartialDerivative{2}(2, (3, 2))
        @test string(pd) == "∂⁵f₂ / ∂x₁³∂x₂²"
    end
end