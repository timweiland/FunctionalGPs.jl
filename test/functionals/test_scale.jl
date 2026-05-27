using FunctionalGPs
using AbstractGPs
using KernelFunctions
using KernelFunctions: ScaleTransform
using LinearAlgebra

@testset "ScaledLinearFunctional" begin
    δ1 = EvaluationFunctional(rand(10))
    δ2 = EvaluationFunctional(rand(10))

    @testset "construction" begin
        @test 2.0 * δ1 isa ScaledLinearFunctional
        @test -1.0 * δ1 isa ScaledLinearFunctional
        @test 1 * δ1 === δ1
        @test output_shape(2.0 * δ1) == output_shape(δ1)
    end

    @testset "nested scaling flattens" begin
        @test 3.0 * (2.0 * δ1) isa ScaledLinearFunctional
        scaled = 3.0 * (2.0 * δ1)
        @test scaled.scalar == 6.0
        @test scaled.linfctl === δ1
    end

    @testset "unary minus" begin
        @test (-δ1) isa ScaledLinearFunctional
        @test (-δ1).scalar == -1
    end

    @testset "subtraction" begin
        diff = δ1 - δ2
        @test diff isa SumLinearFunctional{2}
    end

    @testset "show" begin
        @test occursin("2.0", string(2.0 * δ1))
    end
end

@testset "ScaledLinearFunctional through kernel pipeline" begin
    k = HalfIntegerMaternKernel(2, [1.0])
    X = collect(0.0:0.2:1.0)
    δ = EvaluationFunctional(X)

    @testset "scaled functional produces scaled covariance" begin
        K_base = Matrix(δ(δ(k)))
        K_scaled = Matrix((2.0 * δ)((2.0 * δ)(k)))
        @test K_scaled ≈ 4.0 * K_base
    end

    @testset "negative scaling" begin
        K_base = Matrix(δ(δ(k)))
        K_neg = Matrix((-1.0 * δ)((-1.0 * δ)(k)))
        @test K_neg ≈ K_base
    end

    @testset "subtraction of evaluation functionals" begin
        δ0 = EvaluationFunctional([0.0])
        δ1 = EvaluationFunctional([1.0])
        diff = δ0 - δ1
        K = Matrix(diff(diff(k)))
        @test size(K) == (1, 1)
        # Var(f(0) - f(1)) = k(0,0) - k(0,1) - k(1,0) + k(1,1)
        k00 = only(Matrix(δ0(δ0(k))))
        k01 = only(Matrix(δ0(δ1(k))))
        k10 = only(Matrix(δ1(δ0(k))))
        k11 = only(Matrix(δ1(δ1(k))))
        @test only(K) ≈ k00 - k01 - k10 + k11
    end

    @testset "scaled functional with ZeroMean" begin
        f = GP(k)
        @test (2.0 * δ)(f.mean) == zeros(length(X))
    end

    @testset "scaled functional with stacked crosscov" begin
        δ2 = EvaluationFunctional(rand(4))
        stacked = StackedPVCrosscov([δ(k), δ2(k)])
        result = (2.0 * δ)(stacked)
        @test result ≈ 2.0 * δ(stacked)
    end

    @testset "scaled functional with ConstantScaledPVCrosscov" begin
        pv = 3.0 * δ(k)
        result = (2.0 * δ)(pv)
        @test result ≈ 6.0 * δ(δ(k))
    end
end

@testset "Operator subtraction" begin
    ∂1 = PartialDerivative((1,))
    ∂2 = PartialDerivative((2,))

    @testset "unary minus" begin
        @test (-∂1) isa ConstantScaledLinearFunctionOperator
        @test FunctionalGPs.scale(-∂1) == -1
    end

    @testset "binary subtraction" begin
        diff = ∂2 - ∂1
        @test diff isa SumLinearFunctionOperator{2}
    end

    @testset "subtraction with scaling" begin
        op = ∂2 - 0.5 * ∂1
        @test op isa SumLinearFunctionOperator{2}
    end
end

@testset "LinearlyScaledKernel" begin
    k = SqExponentialKernel()
    X = collect(0.0:0.2:1.0)

    @testset "negative scaled operator on kernel" begin
        op = (-0.5) * PartialDerivative((1,))
        result = op(k)
        @test result isa LinearlyScaledKernel
        @test result.scalar == -0.5
    end

    @testset "positive scaled operator uses ScaledKernel" begin
        op = 2.0 * PartialDerivative((1,))
        result = op(k)
        @test result isa KernelFunctions.ScaledKernel
    end

    @testset "sum of operators with negative scale" begin
        op = PartialDerivative((2,)) - 0.5 * PartialDerivative((1,))
        result = op(k)
        @test result isa KernelFunctions.KernelSum
    end

    @testset "full pipeline: scaled operator through functional" begin
        op = PartialDerivative((2,)) - 0.5 * PartialDerivative((1,))
        ℒ = EvaluationFunctional(X) ∘ op
        K = Matrix(ℒ(ℒ(k)))
        @test size(K) == (length(X), length(X))
        @test K ≈ K'
    end

    @testset "full pipeline with ScaleTransform" begin
        k_scaled = SqExponentialKernel() ∘ ScaleTransform(2.0)
        op = PartialDerivative((2,)) - 0.5 * PartialDerivative((1,))
        ℒ = EvaluationFunctional(X) ∘ op
        K = Matrix(ℒ(ℒ(k_scaled)))
        @test size(K) == (length(X), length(X))
        @test K ≈ K'
    end

    @testset "full pipeline with Matern kernel" begin
        k_mat = HalfIntegerMaternKernel(2, [1.0])
        op = PartialDerivative((1,)) - 0.5 * PartialDerivative((1,))
        ℒ = EvaluationFunctional(X) ∘ op
        K = Matrix(ℒ(ℒ(k_mat)))
        # (1 - 0.5)∂₁ = 0.5∂₁, so K should be 0.25 * K_deriv
        ℒ_full = EvaluationFunctional(X) ∘ PartialDerivative((1,))
        K_full = Matrix(ℒ_full(ℒ_full(k_mat)))
        @test K ≈ 0.25 * K_full
    end

    @testset "stacked functional with negative-scaled kernel" begin
        op = PartialDerivative((1,)) - 2.0 * PartialDerivative((1,))
        ℒ_eval = EvaluationFunctional(X)
        ℒ_deriv = ℒ_eval ∘ PartialDerivative((1,))
        stacked = StackedLinearFunctional(ℒ_eval, ℒ_deriv)
        # op = -1 * ∂₁, so op(k) = LinearlyScaledKernel(∂₁(k), -1)
        K = Matrix(stacked(stacked(op(k))))
        @test size(K) == (2 * length(X), 2 * length(X))
    end

    @testset "integral functional with LinearlyScaledKernel" begin
        k_mat = HalfIntegerMaternKernel(2, [1.0])
        domains = intervals_from_endpoints(range(0.0, 1.0; length = 5))
        op = (-1.0) * PartialDerivative((1,))
        dk = op(k_mat)  # LinearlyScaledKernel
        ℒ = VectorizedLebesgueIntegral(domains)
        pv = ℒ(dk)
        @test pv isa ConstantScaledPVCrosscov
    end
end
