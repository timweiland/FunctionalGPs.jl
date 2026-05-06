using FunctionalGPs
using FunctionalGPs.Notation
using AbstractGPs
using Test

@testset "Notation" begin
    X = [0.0, 0.5, 1.0]

    @testset "δ" begin
        @test δ(X) isa EvaluationFunctional
        @test δ(X).X == X
        @test δ(0.5) isa EvaluationFunctional
        @test δ(0.5).X == [0.5]
    end

    @testset "∂" begin
        op = ∂(1)
        @test op isa PartialDerivative
        @test op.multi_idx == (1,)
        @test op.order == 1
    end

    @testset "∫" begin
        d = Interval(0.0, 1.0)
        @test ∫(d) isa VectorizedLebesgueIntegral
        @test ∫([d]) isa VectorizedLebesgueIntegral
        @test ∫(d, Interval(1.0, 2.0)) isa VectorizedLebesgueIntegral
    end

    @testset "Composition reads naturally inside FunctionalGaussian" begin
        k = WendlandKernel(1, 3, 8 // 10)
        f = GP(k)
        Xd = [0.25, 0.75]
        # The actual slide-form construction.
        fg = FunctionalGaussian(
            f;
            y = δ(X),
            dy = δ(Xd) ∘ ∂(1),
            q = ∫([Interval(0.0, 1.0)]),
        )
        @test keys(fg) == (:y, :dy, :q)
        @test length(fg) == length(X) + length(Xd) + 1
    end
end
