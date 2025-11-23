using GaussPDE
using AbstractGPs
using KernelFunctions
import GaussPDE: randvar_batch_size, randvar_arg, randproc_arg
import KernelFunctions: kernelmatrix
import LinearAlgebra: eigvals

@testset "StackedLinearFunctional" begin
    # Setup test kernels and functionals
    k = SqExponentialKernel()
    k_integrable = WendlandKernel(1, 3)  # For tests requiring integration

    # Create evaluation functionals at different points
    X1 = [0.0, 0.5, 1.0]
    X2 = [1.5, 2.0]
    δ1 = EvaluationFunctional(X1)
    δ2 = EvaluationFunctional(X2)

    # Create integral functionals
    domains1 = [Interval(0.0, 1.0), Interval(1.0, 2.0)]
    domains2 = [Interval(2.0, 3.0)]
    ℒ1 = VectorizedLebesgueIntegral(domains1)
    ℒ2 = VectorizedLebesgueIntegral(domains2)

    @testset "Constructor" begin
        # Test vector constructor
        stacked = StackedLinearFunctional([δ1, δ2])
        @test length(stacked) == 2
        @test stacked[1] == δ1
        @test stacked[2] == δ2

        # Test varargs constructor
        stacked_varargs = StackedLinearFunctional(δ1, δ2)
        @test length(stacked_varargs) == 2

        # Test that empty functionals throws error
        @test_throws ArgumentError StackedLinearFunctional(AbstractLinearFunctional[])

        # Test with mixed functional types
        mixed_stack = StackedLinearFunctional([δ1, ℒ1])
        @test length(mixed_stack) == 2
    end

    @testset "output_shape" begin
        stacked = StackedLinearFunctional([δ1, δ2])
        shapes = output_shape(stacked)
        @test shapes == (size(X1), size(X2))

        mixed_stack = StackedLinearFunctional([δ1, ℒ1])
        mixed_shapes = output_shape(mixed_stack)
        @test mixed_shapes == (size(X1), size(domains1))
    end

    @testset "Iterator interface" begin
        stacked = StackedLinearFunctional([δ1, δ2, ℒ1])
        collected = collect(stacked)
        @test collected == [δ1, δ2, ℒ1]
        @test length(stacked) == 3
    end

    @testset "Application to Kernel (arg=2)" begin
        stacked = StackedLinearFunctional([δ1, δ2])

        # Apply to kernel with arg=2 (default)
        pv = stacked(k; arg=2)

        @test pv isa StackedPVCrosscov
        @test length(pv.pv_crosscovs) == 2
        @test randvar_arg(pv) == 2

        # Verify each component is correct
        @test pv.pv_crosscovs[1] == δ1(k; arg=2)
        @test pv.pv_crosscovs[2] == δ2(k; arg=2)
    end

    @testset "Application to Kernel (arg=1)" begin
        stacked = StackedLinearFunctional([δ1, δ2])

        # Apply to kernel with arg=1
        pv = stacked(k; arg=1)

        @test pv isa StackedPVCrosscov
        @test length(pv.pv_crosscovs) == 2
        @test randvar_arg(pv) == 1

        # Verify each component is correct
        @test pv.pv_crosscovs[1] == δ1(k; arg=1)
        @test pv.pv_crosscovs[2] == δ2(k; arg=1)
    end

    @testset "Symmetric Block Matrix: Evaluation only" begin
        # This is the main use case: applying stacked functional from both sides
        stacked = StackedLinearFunctional([δ1, δ2])

        # Apply from both sides
        pv1 = stacked(k; arg=1)  # Apply to first argument
        result = stacked(pv1)     # Apply to second argument

        # Verify block structure: should be 2x2 blocks
        # Block (1,1): δ1 applied to both sides
        # Block (1,2): δ1 on arg=1, δ2 on arg=2
        # Block (2,1): δ2 on arg=1, δ1 on arg=2
        # Block (2,2): δ2 applied to both sides

        # Compute expected blocks individually and check they match
        block_11_expected = δ1(δ1(k; arg=1))
        block_12_expected = δ1(δ2(k; arg=1))
        block_21_expected = δ2(δ1(k; arg=1))
        block_22_expected = δ2(δ2(k; arg=1))

        # Extract blocks from result and compare
        # Block (1,1) is result[1:length(X1), 1:length(X1)]
        @test result[1:length(X1), 1:length(X1)] ≈ block_11_expected
        # Block (1,2) is result[1:length(X1), (length(X1)+1):end]
        @test result[1:length(X1), (length(X1)+1):end] ≈ block_12_expected
        # Block (2,1) is result[(length(X1)+1):end, 1:length(X1)]
        @test result[(length(X1)+1):end, 1:length(X1)] ≈ block_21_expected
        # Block (2,2) is result[(length(X1)+1):end, (length(X1)+1):end]
        @test result[(length(X1)+1):end, (length(X1)+1):end] ≈ block_22_expected

        # Verify dimensions
        @test size(result, 1) == length(X1) + length(X2)
        @test size(result, 2) == length(X1) + length(X2)

        # Verify symmetry (kernel is symmetric)
        @test result ≈ result'
    end

    @testset "Symmetric Block Matrix: Mixed functionals" begin
        # Test with evaluation and integral functionals
        # Use WendlandKernel which supports integration
        stacked = StackedLinearFunctional([δ1, ℒ1])

        # Apply from both sides
        pv1 = stacked(k_integrable; arg=1)
        result = stacked(pv1)

        # Compute expected blocks
        block_11 = δ1(δ1(k_integrable; arg=1))              # eval-eval
        block_12 = δ1(ℒ1(k_integrable; arg=1))              # eval-integral
        block_21 = ℒ1(δ1(k_integrable; arg=1))              # integral-eval
        block_22 = ℒ1(ℒ1(k_integrable; arg=1))              # integral-integral

        # Extract blocks from result and compare
        @test result[1:length(X1), 1:length(X1)] ≈ block_11
        @test result[1:length(X1), (length(X1)+1):end] ≈ block_12
        @test result[(length(X1)+1):end, 1:length(X1)] ≈ block_21
        @test result[(length(X1)+1):end, (length(X1)+1):end] ≈ block_22

        # Verify dimensions
        @test size(result, 1) == length(X1) + length(domains1)
        @test size(result, 2) == length(X1) + length(domains1)

        # Verify symmetry
        @test result ≈ result'
    end

    @testset "Three functionals" begin
        # Test with three functionals for larger block matrix
        # Use WendlandKernel for integration support
        stacked = StackedLinearFunctional([δ1, δ2, ℒ2])

        pv1 = stacked(k_integrable; arg=1)
        result = stacked(pv1)

        # Verify it's a 3x3 block matrix
        block_11 = δ1(δ1(k_integrable; arg=1))
        block_12 = δ1(δ2(k_integrable; arg=1))
        block_13 = δ1(ℒ2(k_integrable; arg=1))
        block_21 = δ2(δ1(k_integrable; arg=1))
        block_22 = δ2(δ2(k_integrable; arg=1))
        block_23 = δ2(ℒ2(k_integrable; arg=1))
        block_31 = ℒ2(δ1(k_integrable; arg=1))
        block_32 = ℒ2(δ2(k_integrable; arg=1))
        block_33 = ℒ2(ℒ2(k_integrable; arg=1))

        # Verify each block
        n1 = length(X1)
        n2 = length(X2)
        @test result[1:n1, 1:n1] ≈ block_11
        @test result[1:n1, n1+1:n1+n2] ≈ block_12
        @test result[1:n1, n1+n2+1:end] ≈ block_13
        @test result[n1+1:n1+n2, 1:n1] ≈ block_21
        @test result[n1+1:n1+n2, n1+1:n1+n2] ≈ block_22
        @test result[n1+1:n1+n2, n1+n2+1:end] ≈ block_23
        @test result[n1+n2+1:end, 1:n1] ≈ block_31
        @test result[n1+n2+1:end, n1+1:n1+n2] ≈ block_32
        @test result[n1+n2+1:end, n1+n2+1:end] ≈ block_33

        # Verify dimensions
        @test size(result, 1) == length(X1) + length(X2) + length(domains2)
        @test size(result, 2) == length(X1) + length(X2) + length(domains2)

        # Verify symmetry
        @test result ≈ result'
    end

    @testset "Application to ZeroMean" begin
        stacked = StackedLinearFunctional([δ1, δ2])
        m = ZeroMean()

        result = stacked(m)

        # Should be a stacked zero vector
        expected = vcat(zeros(length(X1)), zeros(length(X2)))
        @test result == expected
        @test length(result) == length(X1) + length(X2)
    end

    @testset "Application to ScaledKernel" begin
        σ² = 2.5
        scaled_k = σ² * k
        stacked = StackedLinearFunctional([δ1, δ2])

        pv1 = stacked(scaled_k; arg=1)
        result = stacked(pv1)

        # Should be scaled version of unscaled result
        pv1_unscaled = stacked(k; arg=1)
        result_unscaled = stacked(pv1_unscaled)

        @test result ≈ σ² * σ² * result_unscaled
    end

    @testset "Application to KernelSum" begin
        k2 = MaternKernel()
        k_sum = k + k2
        stacked = StackedLinearFunctional([δ1, δ2])

        pv1_sum = stacked(k_sum; arg=1)
        result_sum = stacked(pv1_sum)

        # Should equal sum of individual results
        pv1_k = stacked(k; arg=1)
        result_k = stacked(pv1_k)

        pv1_k2 = stacked(k2; arg=1)
        result_k2 = stacked(pv1_k2)

        @test result_sum ≈ result_k + result_k2
    end

    @testset "Asymmetric application (different functionals on each side)" begin
        # Apply different stacked functionals to each side
        # Use WendlandKernel for integration support
        stacked1 = StackedLinearFunctional([δ1, ℒ1])
        stacked2 = StackedLinearFunctional([δ2, ℒ2])

        pv1 = stacked1(k_integrable; arg=1)
        result = stacked2(pv1)

        # This creates a non-symmetric block matrix
        # Block (i,j): stacked2[i] applied to arg=2, stacked1[j] applied to arg=1
        block_11 = δ2(δ1(k_integrable; arg=1))
        block_12 = δ2(ℒ1(k_integrable; arg=1))
        block_21 = ℒ2(δ1(k_integrable; arg=1))
        block_22 = ℒ2(ℒ1(k_integrable; arg=1))

        # Verify each block
        @test result[1:length(X2), 1:length(X1)] ≈ block_11
        @test result[1:length(X2), (length(X1)+1):end] ≈ block_12
        @test result[(length(X2)+1):end, 1:length(X1)] ≈ block_21
        @test result[(length(X2)+1):end, (length(X1)+1):end] ≈ block_22

        # This should be symmetric due to kernel symmetry
        @test result ≈ result'
    end

    @testset "Non-square block matrix (arg=2 application)" begin
        # When we apply to arg=2 first, then apply stacked functional,
        # we get a different orientation
        stacked = StackedLinearFunctional([δ1, δ2])

        pv2 = stacked(k; arg=2)  # Apply to second argument

        # Now apply individual functional to this
        result = δ1(pv2)

        # This should give a horizontal concatenation
        expected = hcat(δ1(δ1(k; arg=2)), δ1(δ2(k; arg=2)))
        @test result ≈ expected
    end

    @testset "Integration with GP workflow" begin
        # Test realistic workflow with Gaussian Process
        # Use WendlandKernel for integration support
        f = GP(k_integrable)

        stacked = StackedLinearFunctional([δ1, ℒ1])

        # Apply to GP's kernel and mean
        pv = stacked(f.kernel; arg=1)
        mean_vec = stacked(f.mean)

        @test pv isa StackedPVCrosscov
        @test length(mean_vec) == length(X1) + length(domains1)

        # Compute covariance matrix
        cov_matrix = stacked(pv)
        @test size(cov_matrix, 1) == size(cov_matrix, 2)
        @test size(cov_matrix, 1) == length(X1) + length(domains1)

        # Covariance matrix should be positive semi-definite (approximately)
        eigenvalues = eigvals(Matrix(cov_matrix))
        @test all(eigenvalues .> -1e-10)  # Allow small numerical errors
    end
end
