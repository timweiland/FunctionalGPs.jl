using FunctionalGPs
using FunctionalGPs: ⊗, derivative
using Test
using LinearAlgebra
using Kronecker: kron

@testset "TensorProductFunctional" begin
    # Setup: Create various linear functionals for testing
    X1 = collect(0.0:0.2:1.0)  # 6 points
    X2 = collect(0.0:0.25:1.0)  # 5 points

    δ1 = EvaluationFunctional(X1)
    δ2 = EvaluationFunctional(X2)

    # Create integral functionals
    intervals1 = intervals_from_endpoints(0:0.25:1)  # 4 intervals
    intervals2 = intervals_from_endpoints(0:0.2:1)   # 5 intervals
    ∫1 = VectorizedLebesgueIntegral(intervals1)
    ∫2 = VectorizedLebesgueIntegral(intervals2)

    # Create kernels for testing
    k1 = WendlandKernel(1, 3, 0.8)
    k2 = WendlandKernel(1, 2, 0.5)
    k = k1 ⊗ k2

    @testset "Construction and Properties" begin
        # Basic construction via ⊗
        ℒ = δ1 ⊗ δ2
        @test ℒ isa TensorProductFunctional{2}
        @test length(factors(ℒ)) == 2
        @test factors(ℒ)[1] === δ1
        @test factors(ℒ)[2] === δ2

        # Mixed types (evaluation ⊗ integral)
        ℒ_mixed = δ1 ⊗ ∫2
        @test ℒ_mixed isa TensorProductFunctional{2}
        @test factors(ℒ_mixed)[1] === δ1
        @test factors(ℒ_mixed)[2] === ∫2

        # Three-way tensor product
        X3 = collect(0.0:0.3:1.0)
        δ3 = EvaluationFunctional(X3)
        ℒ_three = δ1 ⊗ δ2 ⊗ δ3
        @test ℒ_three isa TensorProductFunctional{3}
        @test length(factors(ℒ_three)) == 3

        # output_shape should combine the shapes
        @test output_shape(δ1 ⊗ δ2) == (length(X1), length(X2))
        @test output_shape(δ1 ⊗ ∫2) == (length(X1), length(intervals2))
    end

    @testset "Type Stability" begin
        # Test that construction is type-stable
        @test @inferred(δ1 ⊗ δ2) isa TensorProductFunctional
        @test @inferred(δ1 ⊗ ∫2) isa TensorProductFunctional

        # Test that factors retrieval is type-stable
        ℒ = δ1 ⊗ δ2
        @test @inferred(factors(ℒ)) isa Tuple

        # output_shape returns a Tuple (exact type depends on iterator flattening)
        @test output_shape(ℒ) isa Tuple
    end

    @testset "Application to KernelTensorProduct → TensorProductCrosscov" begin
        ℒ = δ1 ⊗ δ2

        # Apply to kernel
        pv = ℒ(k)
        @test pv isa TensorProductCrosscov{2}
        @test length(factors(pv)) == 2

        # Verify the components match individual applications
        @test pv.factors[1] == δ1(k1)
        @test pv.factors[2] == δ2(k2)

        # Mixed types: evaluation ⊗ integral
        ℒ_mixed = δ1 ⊗ ∫2
        pv_mixed = ℒ_mixed(k)
        @test pv_mixed isa TensorProductCrosscov{2}
        @test pv_mixed.factors[1] == δ1(k1)
        @test pv_mixed.factors[2] == ∫2(k2)

        # Test arg parameter
        pv_arg1 = ℒ(k; arg = 1)
        @test pv_arg1 isa TensorProductCrosscov{2}
        @test randvar_arg(pv_arg1) == 1
    end

    @testset "Application to TensorProductCrosscov → Kronecker Matrix" begin
        ℒ = δ1 ⊗ δ2

        # One-sided application
        pv = ℒ(k)

        # Two-sided application should produce Kronecker product
        K = ℒ(pv)

        # Compare with manual Kronecker product
        K1 = δ1(δ1(k1))
        K2 = δ2(δ2(k2))
        K_manual = kron(K2, K1)

        @test size(K) == size(K_manual)
        @test K ≈ K_manual
    end

    @testset "Mixed Functionals: eval ⊗ integral" begin
        # Line integral scenario: evaluate at X1, integrate over intervals2
        ℒ = δ1 ⊗ ∫2

        # One-sided application
        pv = ℒ(k)
        @test pv isa TensorProductCrosscov{2}

        # Two-sided application
        K = ℒ(pv)

        # Compare with manual computation
        K1 = δ1(δ1(k1))
        K2 = ∫2(∫2(k2))
        K_manual = kron(K2, K1)

        @test size(K) == size(K_manual)
        @test K ≈ K_manual
    end

    @testset "Numerical Correctness: Full Integration Test" begin
        # Create a specific setup where we can verify numerically
        X_eval = [0.3, 0.6]
        intervals_int = [Interval(0.0, 0.5), Interval(0.5, 1.0)]

        δ_x = EvaluationFunctional(X_eval)
        ∫_y = VectorizedLebesgueIntegral(intervals_int)

        ℒ = δ_x ⊗ ∫_y

        # Apply to tensor product kernel
        pv = ℒ(k)
        K = ℒ(pv)

        # Compute manually using individual operations
        K_x = δ_x(δ_x(k1))  # Evaluation kernel matrix for k1
        K_y = ∫_y(∫_y(k2))  # Integral kernel matrix for k2
        K_expected = kron(K_y, K_x)

        @test K ≈ K_expected
    end

    @testset "Dimension Mismatch Error" begin
        # Create a 2-factor functional but apply to 3-factor kernel
        k3 = WendlandKernel(1, 1, 0.3)
        k_3d = k1 ⊗ k2 ⊗ k3

        ℒ_2d = δ1 ⊗ δ2

        @test_throws ArgumentError ℒ_2d(k_3d)
    end

    @testset "Composition with PartialDerivative" begin
        # Test that (δ₁ ⊗ ∫₂) ∘ ∂₂ works correctly
        # This represents: evaluate at x, integrate the y-derivative over y
        ℒ = δ1 ⊗ ∫2
        ∂_y = PartialDerivative((0, 1))  # Derivative in second dimension

        # Compose and apply
        ℒ_∂ = ℒ ∘ ∂_y
        pv = ℒ_∂(k)

        # The derivative should distribute to the y-component
        # (δ₁ ⊗ ∫₂) ∘ ∂₂(k₁ ⊗ k₂) = δ₁(k₁) ⊗ ∫₂(∂k₂)
        dk2 = derivative(k2, 0, 1)
        expected_pv1 = δ1(k1)
        expected_pv2 = ∫2(dk2)

        @test pv isa TensorProductCrosscov{2}
        @test pv.factors[1] == expected_pv1

        # For the second factor, structural equality may not hold due to
        # different construction paths, so we compare numerical results
        K2_actual = ∫2(pv.factors[2])
        K2_expected = ∫2(expected_pv2)
        @test K2_actual ≈ K2_expected
    end

    @testset "Integral × Integral" begin
        # Both dimensions are integrals
        ℒ = ∫1 ⊗ ∫2

        # Apply to kernel
        pv = ℒ(k)
        @test pv isa TensorProductCrosscov{2}

        # Two-sided application
        K = ℒ(pv)

        # Compare with manual computation
        K1 = ∫1(∫1(k1))
        K2 = ∫2(∫2(k2))
        K_manual = kron(K2, K1)

        @test K ≈ K_manual
    end

    @testset "Comparison with FactorizedBoxDomains" begin
        # The TensorProductFunctional with all integrals should give
        # the same result as using FactorizedBoxDomains

        # Setup matching domains
        ∫_tensor = ∫1 ⊗ ∫2
        ∫_box = VectorizedLebesgueIntegral(intervals1 ⊗ intervals2)

        # Apply to kernel
        pv_tensor = ∫_tensor(k)
        pv_box = ∫_box(k)

        # Both should be TensorProductCrosscov with matching factors
        @test pv_tensor isa TensorProductCrosscov{2}
        @test pv_box isa TensorProductCrosscov{2}
        @test pv_tensor.factors[1] == pv_box.factors[1]
        @test pv_tensor.factors[2] == pv_box.factors[2]

        # Two-sided application should match
        K_tensor = ∫_tensor(pv_tensor)
        K_box = ∫_box(pv_box)

        @test K_tensor ≈ K_box
    end
end
