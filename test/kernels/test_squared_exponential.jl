using ForwardDiff
using KernelFunctions: SqExponentialKernel, ScaleTransform
using ToeplitzMatrices: SymmetricToeplitz

# ============================================================================
# Hermite polynomial tests
# ============================================================================

@testset "Probabilist Hermite polynomials" begin
    # Test known values
    He0 = FunctionalGPs.probabilist_hermite(0)
    He1 = FunctionalGPs.probabilist_hermite(1)
    He2 = FunctionalGPs.probabilist_hermite(2)
    He3 = FunctionalGPs.probabilist_hermite(3)
    He4 = FunctionalGPs.probabilist_hermite(4)

    # He_0(x) = 1
    @test He0(0.0) == 1.0
    @test He0(1.5) == 1.0

    # He_1(x) = x
    @test He1(0.0) == 0.0
    @test He1(2.0) == 2.0

    # He_2(x) = x^2 - 1
    @test He2(0.0) == -1.0
    @test He2(2.0) == 3.0

    # He_3(x) = x^3 - 3x
    @test He3(0.0) == 0.0
    @test He3(2.0) == 2.0

    # He_4(x) = x^4 - 6x^2 + 3
    @test He4(0.0) == 3.0
    @test He4(2.0) == -5.0  # 16 - 24 + 3 = -5

    # Test recurrence relation: He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)
    x = 1.7
    for n in 1:6
        He_n = FunctionalGPs.probabilist_hermite(n)
        He_n1 = FunctionalGPs.probabilist_hermite(n + 1)
        He_nm1 = FunctionalGPs.probabilist_hermite(n - 1)
        @test He_n1(x) ≈ x * He_n(x) - n * He_nm1(x) atol = 1.0e-12
    end

    # Test derivative property: d^n/dx^n[exp(-x²/2)] = (-1)^n * He_n(x) * exp(-x²/2)
    f = x -> exp(-x^2 / 2)
    for n in 0:5
        He_n = FunctionalGPs.probabilist_hermite(n)
        # Use autodiff to compute n-th derivative
        df = f
        for _ in 1:n
            df_prev = df
            df = t -> ForwardDiff.derivative(df_prev, t)
        end
        for x in [-1.5, 0.0, 0.8, 2.1]
            expected = (-1)^n * He_n(x) * exp(-x^2 / 2)
            @test df(x) ≈ expected atol = 1.0e-8 rtol = 1.0e-6
        end
    end
end

# ============================================================================
# SE kernel derivative tests
# ============================================================================

function _autodiff_se_derivative(k, x, y, n, m)
    if n == 0 && m == 0
        return k(x, y)
    elseif n > 0
        return ForwardDiff.derivative(t -> _autodiff_se_derivative(k, t, y, n - 1, m), x)
    else
        return ForwardDiff.derivative(t -> _autodiff_se_derivative(k, x, t, n, m - 1), y)
    end
end

@testset "SqExponentialKernel derivatives" begin
    @testset "Unit scale" begin
        k = SqExponentialKernel()
        x = 0.3
        y = 1.1
        orders = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]

        for (n, m) in orders
            D = FunctionalGPs.derivative(k, n, m)
            expected = _autodiff_se_derivative(k, x, y, n, m)
            @test D(x, y) ≈ expected atol = 1.0e-9 rtol = 1.0e-7
        end

        # Odd derivative at coincident point should be zero
        D_odd = FunctionalGPs.derivative(k, 1, 0)
        @test D_odd(0.5, 0.5) == 0.0

        # Even derivatives are stationary -> Toeplitz matrix
        D_even = FunctionalGPs.derivative(k, 2, 0)
        @test kernel_structure(D_even.derivative_kernel) isa StationaryKernelTrait
        grid = range(0.0; stop = 1.0, length = 6)
        mat = kernel_evaluate_evaluate(D_even.derivative_kernel, grid)
        @test mat isa SymmetricToeplitz

        # Odd derivatives are signed-stationary
        D_odd = FunctionalGPs.derivative(k, 1, 0)
        @test kernel_structure(D_odd.derivative_kernel) isa SignedStationaryKernelTrait
    end

    @testset "With ScaleTransform (lengthscale)" begin
        ℓ = 0.7
        k = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)
        x = 0.25
        y = 0.9
        orders = [(1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (2, 1)]

        for (n, m) in orders
            D = FunctionalGPs.derivative(k, n, m)
            expected = _autodiff_se_derivative(k, x, y, n, m)
            @test D(x, y) ≈ expected atol = 1.0e-9 rtol = 1.0e-7
        end

        # Verify scale extraction
        @test FunctionalGPs._se_scale(k) ≈ 1 / ℓ

        # Even derivative produces Toeplitz
        D_even = FunctionalGPs.derivative(k, 2, 0)
        grid = range(0.0; stop = 1.0, length = 6)
        mat = kernel_evaluate_evaluate(D_even.derivative_kernel, grid)
        @test mat isa SymmetricToeplitz
    end
end

# ============================================================================
# Trait dispatch tests
# ============================================================================

@testset "SqExponentialKernel trait dispatch" begin
    k_unit = SqExponentialKernel()
    k_scaled = SqExponentialKernel() ∘ ScaleTransform(2.0)

    @test kernel_structure(k_unit) isa StationaryKernelTrait
    @test kernel_structure(k_scaled) isa StationaryKernelTrait

    # Test stationary kernel spec
    spec_unit = stationary_kernel_spec(k_unit, Float64)
    @test spec_unit isa StationaryKernelSpec
    @test spec_unit.scales ≈ [1.0]
    @test spec_unit.radial_map(0.0) ≈ 1.0
    @test spec_unit.radial_map(1.0) ≈ exp(-0.5)

    spec_scaled = stationary_kernel_spec(k_scaled, Float64)
    @test spec_scaled isa StationaryKernelSpec
    @test spec_scaled.scales ≈ [2.0]
end

# ============================================================================
# Cross-covariance matrix tests
# ============================================================================

@testset "SqExponentialKernel cross-covariance" begin
    k = SqExponentialKernel() ∘ ScaleTransform(1.5)

    X_left = collect(range(0.0; stop = 0.8, length = 5))
    X_right = collect(range(0.1; stop = 0.9, length = 5))

    lazy_cross = kernel_evaluate_evaluate(k, X_left, X_right)
    dense_cross = kernelmatrix(k, X_left, X_right)

    @test lazy_cross ≈ dense_cross atol = 1.0e-9 rtol = 1.0e-7
end
