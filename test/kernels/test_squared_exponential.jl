using ForwardDiff
using KernelFunctions: SqExponentialKernel, ScaleTransform, ARDTransform, KernelTensorProduct
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

# ============================================================================
# Multi-dimensional derivative tests
# ============================================================================

function _autodiff_se_derivative_nd(k, x, y, α, β)
    # α, β are tuples of derivative orders per dimension
    # Recursively compute mixed partial derivatives via ForwardDiff
    D = length(α)
    @assert D == length(β) == length(x) == length(y)

    result = k(x, y)

    # Apply derivatives w.r.t. x components
    for (i, order) in enumerate(α)
        for _ in 1:order
            x_val = x[i]
            result = ForwardDiff.derivative(
                t -> begin
                    x_new = collect(x)
                    x_new[i] = t
                    k(x_new, y)
                end, x_val
            )
            # Update for next iteration - need fresh closure each time
            x_i = x[i]
            result_prev = result
            k_partial = (x_test, y_test) -> begin
                # This is getting complex, let's simplify
                return result_prev
            end
        end
    end

    # Actually, let's use a simpler recursive approach
    return _autodiff_partial_recursive(k, collect(x), collect(y), collect(α), collect(β))
end

function _autodiff_partial_recursive(k, x, y, α, β)
    # Find first nonzero derivative order
    for i in eachindex(α)
        if α[i] > 0
            α_new = copy(α)
            α_new[i] -= 1
            return ForwardDiff.derivative(
                t -> begin
                    # Promote all elements to the same type as t
                    T = typeof(t)
                    x_new = Vector{T}(undef, length(x))
                    for j in eachindex(x)
                        x_new[j] = j == i ? t : T(x[j])
                    end
                    return _autodiff_partial_recursive(k, x_new, y, α_new, β)
                end,
                x[i],
            )
        end
    end
    for i in eachindex(β)
        if β[i] > 0
            β_new = copy(β)
            β_new[i] -= 1
            return ForwardDiff.derivative(
                t -> begin
                    T = typeof(t)
                    y_new = Vector{T}(undef, length(y))
                    for j in eachindex(y)
                        y_new[j] = j == i ? t : T(y[j])
                    end
                    return _autodiff_partial_recursive(k, x, y_new, α, β_new)
                end,
                y[i],
            )
        end
    end
    # All derivatives done
    return k(x, y)
end

@testset "SE tensor product decomposition" begin
    # Test se_tensor_product construction
    k2 = FunctionalGPs.se_tensor_product(2)
    @test k2 isa KernelTensorProduct
    @test length(k2.kernels) == 2

    # With scales
    k2_scaled = FunctionalGPs.se_tensor_product([2.0, 3.0])
    @test k2_scaled isa KernelTensorProduct
    @test length(k2_scaled.kernels) == 2

    # Tensor product should give same result as base SE kernel
    k_base = SqExponentialKernel()
    x = [0.3, 0.5]
    y = [0.7, 0.2]
    @test k2(x, y) ≈ k_base(x, y) atol = 1.0e-12
end

@testset "Multi-D SE derivatives via PartialDerivative" begin
    @testset "2D unit scale" begin
        k = SqExponentialKernel()
        x = [0.3, 0.5]
        y = [0.7, 0.2]

        # ∂/∂y₁ (default arg=2 means second argument)
        ∂y1 = PartialDerivative((1, 0))
        dk = ∂y1(k)
        @test dk isa KernelTensorProduct
        expected = _autodiff_partial_recursive(k, x, y, [0, 0], [1, 0])
        @test dk(x, y) ≈ expected atol = 1.0e-8 rtol = 1.0e-6

        # ∂/∂y₂
        ∂y2 = PartialDerivative((0, 1))
        dk2 = ∂y2(k)
        expected2 = _autodiff_partial_recursive(k, x, y, [0, 0], [0, 1])
        @test dk2(x, y) ≈ expected2 atol = 1.0e-8 rtol = 1.0e-6

        # ∂²/∂y₁∂y₂
        ∂y1y2 = PartialDerivative((1, 1))
        dk12 = ∂y1y2(k)
        expected12 = _autodiff_partial_recursive(k, x, y, [0, 0], [1, 1])
        @test dk12(x, y) ≈ expected12 atol = 1.0e-8 rtol = 1.0e-6

        # ∂²/∂y₁²
        ∂y1y1 = PartialDerivative((2, 0))
        dk11 = ∂y1y1(k)
        expected11 = _autodiff_partial_recursive(k, x, y, [0, 0], [2, 0])
        @test dk11(x, y) ≈ expected11 atol = 1.0e-8 rtol = 1.0e-6
    end

    @testset "2D with ARDTransform" begin
        ℓ1, ℓ2 = 0.5, 1.2
        k = SqExponentialKernel() ∘ ARDTransform([1 / ℓ1, 1 / ℓ2])
        x = [0.2, 0.8]
        y = [0.6, 0.3]

        # ∂/∂y₁ (default arg=2)
        ∂y1 = PartialDerivative((1, 0))
        dk = ∂y1(k)
        expected = _autodiff_partial_recursive(k, x, y, [0, 0], [1, 0])
        @test dk(x, y) ≈ expected atol = 1.0e-8 rtol = 1.0e-6

        # ∂²/∂y₁∂y₂
        ∂y1y2 = PartialDerivative((1, 1))
        dk12 = ∂y1y2(k)
        expected12 = _autodiff_partial_recursive(k, x, y, [0, 0], [1, 1])
        @test dk12(x, y) ≈ expected12 atol = 1.0e-8 rtol = 1.0e-6

        # Higher order: ∂³/∂y₁²∂y₂
        ∂y1y1y2 = PartialDerivative((2, 1))
        dk112 = ∂y1y1y2(k)
        expected112 = _autodiff_partial_recursive(k, x, y, [0, 0], [2, 1])
        @test dk112(x, y) ≈ expected112 atol = 1.0e-7 rtol = 1.0e-5
    end

    @testset "3D" begin
        k = SqExponentialKernel()
        x = [0.1, 0.4, 0.7]
        y = [0.3, 0.2, 0.9]

        # ∂/∂y₃ (default arg=2)
        ∂y3 = PartialDerivative((0, 0, 1))
        dk = ∂y3(k)
        @test dk isa KernelTensorProduct
        @test length(dk.kernels) == 3
        expected = _autodiff_partial_recursive(k, x, y, [0, 0, 0], [0, 0, 1])
        @test dk(x, y) ≈ expected atol = 1.0e-8 rtol = 1.0e-6

        # ∂²/∂y₁∂y₃
        ∂y1y3 = PartialDerivative((1, 0, 1))
        dk13 = ∂y1y3(k)
        expected13 = _autodiff_partial_recursive(k, x, y, [0, 0, 0], [1, 0, 1])
        @test dk13(x, y) ≈ expected13 atol = 1.0e-8 rtol = 1.0e-6
    end
end
