using ForwardDiff
using ToeplitzMatrices: SymmetricToeplitz

function _autodiff_matern_derivative(k, x, y, n, m)
    if n == 0 && m == 0
        return k(x, y)
    elseif n > 0
        return ForwardDiff.derivative(t -> _autodiff_matern_derivative(k, t, y, n - 1, m), x)
    else
        return ForwardDiff.derivative(t -> _autodiff_matern_derivative(k, x, t, n, m - 1), y)
    end
end

@testset "HalfIntegerMatern derivatives" begin
    k = HalfIntegerMaternKernel(2, [0.8])
    x = 0.3
    y = 1.1
    orders = [(1, 0), (0, 1), (2, 0), (1, 1), (3, 0)]

    for (n, m) in orders
        D = FunctionalGPs.derivative(k, n, m)
        expected = _autodiff_matern_derivative(k, x, y, n, m)
        @test D(x, y) ≈ expected atol = 1.0e-9 rtol = 1.0e-7
    end

    D_odd = FunctionalGPs.derivative(k, 1, 0)
    @test D_odd(0.5, 0.5) == 0

    D_even = FunctionalGPs.derivative(k, 2, 0)
    @test kernel_structure(D_even.derivative_kernel) isa StationaryKernelTrait
    grid = range(0.0, stop = 1.0, length = 6)
    mat = kernel_evaluate_evaluate(D_even.derivative_kernel, grid)
    @test mat isa SymmetricToeplitz

    X_left = collect(range(0.0, stop = 0.8, length = 5))
    X_right = collect(range(0.1, stop = 0.9, length = 5))
    lazy_cross = kernel_evaluate_evaluate(k, X_left, X_right)
    dense_cross = kernelmatrix(k, X_left, X_right)
    @test lazy_cross ≈ dense_cross atol = 1.0e-9 rtol = 1.0e-7
end

@testset "Scalar lengthscale routes through stationary spec" begin
    # Regression: building the kernel with a *scalar* lengthscale (allowed per
    # the constructor docstring) used to break point-evaluation assembly —
    # `collect(scale ./ scalar)` yields a 0-dim array, which the
    # `StationaryKernelSpec` constructor (TS <: AbstractVector) rejects.
    # A scalar lengthscale `ℓ` must behave exactly like the 1-vector `[ℓ]`.
    X = collect(0.0:0.1:1.0)
    L = EvaluationFunctional(X)

    k_scalar = HalfIntegerMaternKernel(2, 0.3)
    k_vector = HalfIntegerMaternKernel(2, [0.3])

    M_scalar = Matrix(L(L(k_scalar)))
    @test M_scalar ≈ Matrix(L(L(k_vector)))
    @test M_scalar ≈ kernelmatrix(k_scalar, X)

    # Even/odd derivative kernels reuse `base_spec.scales`, so they must work
    # with scalar lengthscales too.
    for (n, m) in ((2, 0), (1, 0))
        D_s = FunctionalGPs.derivative(k_scalar, n, m).derivative_kernel
        D_v = FunctionalGPs.derivative(k_vector, n, m).derivative_kernel
        @test Matrix(kernel_evaluate_evaluate(D_s, X)) ≈
            Matrix(kernel_evaluate_evaluate(D_v, X))
    end

    # AD must still flow through a scalar lengthscale carried as a Dual
    # (the spec wraps it in a vector without converting to the element type).
    g = ForwardDiff.derivative(ℓ -> sum(Matrix(L(L(HalfIntegerMaternKernel(2, ℓ))))), 0.3)
    @test isfinite(g) && !iszero(g)
end
