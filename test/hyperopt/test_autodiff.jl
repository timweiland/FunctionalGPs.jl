using ForwardDiff
using FiniteDifferences
using LinearAlgebra
using KernelFunctions: SqExponentialKernel, ScaleTransform, ⊗, with_lengthscale, Matern52Kernel
using AbstractGPs: mean
using ChainRulesCore

# ============================================================================
# Helpers
# ============================================================================

"""
    _gp_nll(K, y)

Compute the GP negative log-likelihood given Gram matrix `K` and observations `y`.
"""
function _gp_nll(K, y)
    G = Symmetric(K)
    C = cholesky(G)
    α = C \ y
    N = length(y)
    return oftype(α[1], 0.5) * (dot(y, α) + logdet(C) + N * log(2π))
end

"""
    _nll_from_gram(K, y; noise)

Add diagonal noise and compute the NLL. Shared by most tests.
"""
function _nll_from_gram(K, y; noise = 0.01)
    K = copy(K)
    K[diagind(K)] .+= noise
    return gp_nll(K, y)
end

"""
    _check_ad_gradient(f, θ0; rtol)

Assert ForwardDiff gradient matches FiniteDifferences.
"""
function _check_ad_gradient(f, θ0; rtol = 1.0e-5)
    ad = ForwardDiff.gradient(f, θ0)
    fd = FiniteDifferences.grad(central_fdm(5, 1), f, θ0)[1]
    return @test ad ≈ fd rtol = rtol
end

# ============================================================================
# Kernel and input factories
# ============================================================================

KERNEL_FACTORIES_1D = [
    ("Matern-1/2", θ -> HalfIntegerMaternKernel(0, exp.(θ))),
    ("Matern-3/2", θ -> HalfIntegerMaternKernel(1, exp.(θ))),
    ("Matern-5/2", θ -> HalfIntegerMaternKernel(2, exp.(θ))),
    ("SE-scaled", θ -> SqExponentialKernel() ∘ ScaleTransform(exp(θ[1]))),
]

MATERN_FACTORIES_1D = [
    ("Matern-3/2", θ -> HalfIntegerMaternKernel(1, exp.(θ))),
    ("Matern-5/2", θ -> HalfIntegerMaternKernel(2, exp.(θ))),
]

# Kernels smooth enough for derivative observations (exclude Matern-1/2)
SMOOTH_KERNEL_FACTORIES_1D = [
    ("Matern-3/2", θ -> HalfIntegerMaternKernel(1, exp.(θ))),
    ("Matern-5/2", θ -> HalfIntegerMaternKernel(2, exp.(θ))),
    ("SE-scaled", θ -> SqExponentialKernel() ∘ ScaleTransform(exp(θ[1]))),
]

INPUT_TYPES_1D = [
    ("collected", collect(0.0:0.2:1.0)),
    ("range", 0.0:0.2:1.0),
]

DERIVATIVE_CONFIGS = [
    ("∂k/∂x∂y (odd)", (1, 1)),
    ("∂²k/∂x² (even)", (2, 0)),
]

# Shared data
X_1D = collect(0.0:0.2:1.0)
Y_EVAL = sin.(X_1D)
Y_DERIV = cos.(X_1D)

# ============================================================================
# Combinatorial coverage: kernels × input types × observation types
# ============================================================================

@testset "ForwardDiff combinatorial coverage" begin
    @testset "Base kernel: $kname × $iname" for (kname, make_k) in KERNEL_FACTORIES_1D,
            (iname, X) in INPUT_TYPES_1D

        nll(θ) = _nll_from_gram(Matrix(kernel_evaluate_evaluate(make_k(θ), X)), Y_EVAL)
        _check_ad_gradient(nll, [0.0])
    end

    @testset "Derivative: $kname × $dname" for (kname, make_k) in KERNEL_FACTORIES_1D,
            (dname, (n, m)) in DERIVATIVE_CONFIGS

        function nll(θ)
            dk = FunctionalGPs.derivative(make_k(θ), n, m)
            K = Matrix(kernel_evaluate_evaluate(dk.derivative_kernel, X_1D))
            # ∂²k/∂x² can have negative eigenvalues — need large noise
            return _nll_from_gram(K, Y_DERIV; noise = 10.0)
        end
        _check_ad_gradient(nll, [0.0]; rtol = 1.0e-4)
    end

    X_left = collect(0.0:0.3:0.9)
    X_right = collect(0.0:0.2:1.0)

    @testset "Cross-cov: $kname" for (kname, make_k) in KERNEL_FACTORIES_1D
        f(θ) = sum(Matrix(kernel_evaluate_evaluate(make_k(θ), X_left, X_right)))
        _check_ad_gradient(f, [0.0])
    end

    domains = intervals_from_endpoints(range(0.0, 1.0; length = 7))
    y_integ = [cos(d.lower) - cos(d.upper) for d in domains]

    @testset "Integral: $kname" for (kname, make_k) in MATERN_FACTORIES_1D
        nll(θ) =
            _nll_from_gram(Matrix(kernel_integrate_integrate(make_k(θ), domains)), y_integ)
        _check_ad_gradient(nll, [0.0]; rtol = 1.0e-4)
    end

    @testset "StackedFunctional: $kname" for (kname, make_k) in SMOOTH_KERNEL_FACTORIES_1D
        function nll(θ)
            k = make_k(θ)
            ℒ = StackedLinearFunctional(
                EvaluationFunctional(X_1D),
                EvaluationFunctional(X_1D) ∘ PartialDerivative((1,)),
            )
            return _nll_from_gram(Matrix(ℒ(ℒ(k))), [Y_EVAL; Y_DERIV]; noise = 0.1)
        end
        _check_ad_gradient(nll, [0.0]; rtol = 1.0e-4)
    end

    @testset "Noise as parameter" begin
        function nll_noise(θ)
            k = HalfIntegerMaternKernel(2, exp.(θ[1:1]))
            K = Matrix(kernel_evaluate_evaluate(k, X_1D)) + exp(θ[2]) * I
            return gp_nll(K, Y_EVAL)
        end
        _check_ad_gradient(nll_noise, [0.0, -4.0])
    end

    # --- Multi-dimensional ---
    X_2d = [0.0 0.0; 0.3 0.1; 0.6 0.4; 0.9 0.7; 0.2 0.8; 0.5 0.5]
    y_2d = sin.(X_2d[:, 1]) .* cos.(X_2d[:, 2])

    @testset "2D Matern: $kname" for (kname, P) in [("Matern-3/2", 1), ("Matern-5/2", 2)]
        nll(θ) =
            _nll_from_gram(
            Matrix(kernel_evaluate_evaluate(HalfIntegerMaternKernel(P, exp.(θ)), X_2d)),
            y_2d,
        )
        _check_ad_gradient(nll, [0.0, 0.0])
    end

    @testset "2D SE tensor product" begin
        X_vov = [collect(row) for row in eachrow(X_2d)]
        nll(θ) =
            _nll_from_gram(
            Matrix(kernel_evaluate_evaluate(FunctionalGPs.se_tensor_product(exp.(θ)), X_vov)),
            y_2d,
        )
        _check_ad_gradient(nll, [0.0, 0.0])
    end
end

# ============================================================================
# Edge cases not covered by combinatorial sweep
# ============================================================================

@testset "ForwardDiff through functional pipeline" begin
    @testset "EvaluationFunctional double application" begin
        function nll(θ)
            ℒ = EvaluationFunctional(X_1D)
            return _nll_from_gram(Matrix(ℒ(ℒ(HalfIntegerMaternKernel(2, exp.(θ))))), Y_EVAL)
        end
        _check_ad_gradient(nll, [0.0])
    end

    @testset "Derivative functional composition" begin
        function nll(θ)
            ℒ = EvaluationFunctional(X_1D) ∘ PartialDerivative((1,))
            return _nll_from_gram(Matrix(ℒ(ℒ(HalfIntegerMaternKernel(2, exp.(θ))))), Y_DERIV)
        end
        _check_ad_gradient(nll, [0.0])
    end
end

@testset "ForwardDiff through condition_on_observation" begin
    X_test = collect(0.0:0.1:1.0)

    @testset "Posterior mean gradient w.r.t. lengthscale" begin
        function posterior_mean_sum(θ)
            k = HalfIntegerMaternKernel(2, exp.(θ))
            f = GP(k)
            f_post = condition_on_observation(f, X_1D, Y_EVAL; noise = 0.01)
            return sum(mean(f_post(X_test)))
        end

        _check_ad_gradient(posterior_mean_sum, [0.0])
    end
end

# ============================================================================
# Float64 regression: lazy matrix types still work after T<:Real relaxation
# ============================================================================

@testset "Float64 paths unbroken" begin
    @testset "StationaryKernelMatrix" begin
        K = kernel_evaluate_evaluate(HalfIntegerMaternKernel(2, [1.0]), X_1D)
        @test K isa StationaryKernelMatrix
        @test eltype(K) == Float64
        @test size(K) == (length(X_1D), length(X_1D))
        @test Matrix(K) isa Matrix{Float64}
        @test Matrix(K) ≈ kernelmatrix(HalfIntegerMaternKernel(2, [1.0]), X_1D)
    end

    @testset "SignedStationaryKernelMatrix" begin
        dk = FunctionalGPs.derivative(HalfIntegerMaternKernel(2, [1.0]), 1, 0)
        K = kernel_evaluate_evaluate(dk.derivative_kernel, X_1D)
        @test K isa SignedStationaryKernelMatrix
        @test eltype(K) == Float64
    end

    @testset "SE kernel" begin
        K = kernel_evaluate_evaluate(SqExponentialKernel() ∘ ScaleTransform(2.0), X_1D)
        @test K isa StationaryKernelMatrix
        @test eltype(K) == Float64
    end
end

# ============================================================================
# ChainRules: rrule for gp_nll (reverse-mode support)
# ============================================================================

@testset "gp_nll rrule" begin
    N = length(X_1D)
    K_mat = Matrix(kernel_evaluate_evaluate(HalfIntegerMaternKernel(2, [1.0]), X_1D)) + 0.01 * I

    @testset "pullback matches analytic formula" begin
        nll_val, pullback = ChainRulesCore.rrule(gp_nll, K_mat, Y_EVAL)
        @test nll_val ≈ gp_nll(K_mat, Y_EVAL)

        _, ∂K, _ = pullback(1.0)
        ∂K_mat = ∂K isa ChainRulesCore.Thunk ? ChainRulesCore.unthunk(∂K) : ∂K

        C = cholesky(Symmetric(K_mat))
        α = C \ Y_EVAL
        Kinv = C \ Matrix(I, N, N)
        @test ∂K_mat ≈ (Kinv - α * α') / 2 rtol = 1.0e-10
    end

    @testset "ForwardDiff consistency with manual NLL" begin
        nll_exported(θ) =
            gp_nll(
            Matrix(kernel_evaluate_evaluate(HalfIntegerMaternKernel(2, exp.(θ)), X_1D)) +
                0.01 * I,
            Y_EVAL,
        )
        nll_manual(θ) =
            _gp_nll(
            Matrix(kernel_evaluate_evaluate(HalfIntegerMaternKernel(2, exp.(θ)), X_1D)) +
                0.01 * I,
            Y_EVAL,
        )

        θ0 = [0.0]
        @test ForwardDiff.gradient(nll_exported, θ0) ≈
            ForwardDiff.gradient(nll_manual, θ0) rtol = 1.0e-10
    end
end
