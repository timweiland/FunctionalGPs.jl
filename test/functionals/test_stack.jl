using GaussPDE
using Test
using LinearAlgebra
using BlockArrays

@testset "StackedLinearFunctional" begin
    # Setup: Create various linear functionals for testing
    X1 = rand(10)
    X2 = rand(5)
    X3 = rand(7)

    δ1 = EvaluationFunctional(X1)
    δ2 = EvaluationFunctional(X2)
    δ3 = EvaluationFunctional(X3)

    # Create integral functionals for mixed-type testing
    domains_1d = [Interval(0.0, 1.0), Interval(1.0, 2.0)]
    domains_2d = [Interval(0.0, 1.0)]
    ℒ1 = VectorizedLebesgueIntegral(domains_1d)
    ℒ2 = VectorizedLebesgueIntegral(domains_2d)

    # Create a GP for testing
    k = WendlandKernel(1, 3)
    f = GP(k)

    @testset "Construction and Properties" begin
        # Basic construction
        stacked_eval = StackedLinearFunctional([δ1, δ2])
        @test stacked_eval isa StackedLinearFunctional
        @test length(stacked_eval.linfunctionals) == 2

        # Mixed types (heterogeneous)
        stacked_mixed = StackedLinearFunctional([δ1, ℒ1])
        @test stacked_mixed isa StackedLinearFunctional
        @test length(stacked_mixed.linfunctionals) == 2

        # output_shape should sum the lengths
        @test output_shape(stacked_eval) == (length(X1) + length(X2),)

        # Single functional in stack
        stacked_single = StackedLinearFunctional([δ1])
        @test output_shape(stacked_single) == (length(X1),)

        # Three functionals
        stacked_three = StackedLinearFunctional([δ1, δ2, δ3])
        @test output_shape(stacked_three) == (length(X1) + length(X2) + length(X3),)
    end

    @testset "Application to Kernel → StackedPVCrosscov" begin
        stacked_lf = StackedLinearFunctional([δ1, δ2])

        # Apply to kernel with arg=2 (default)
        pv_stack = stacked_lf(f.kernel)
        @test pv_stack isa StackedPVCrosscov
        @test length(pv_stack.pv_crosscovs) == 2
        @test randvar_arg(pv_stack) == 2
        @test randvar_batch_size(pv_stack) == (length(X1) + length(X2),)

        # Apply to kernel with arg=1
        pv_stack_arg1 = stacked_lf(f.kernel, arg = 1)
        @test pv_stack_arg1 isa StackedPVCrosscov
        @test randvar_arg(pv_stack_arg1) == 1

        # Verify the components are correct
        @test pv_stack.pv_crosscovs[1] isa EvaluationPVCrosscov
        @test pv_stack.pv_crosscovs[2] isa EvaluationPVCrosscov

        # Mixed types
        stacked_mixed = StackedLinearFunctional([δ1, ℒ1])
        pv_mixed = stacked_mixed(f.kernel)
        @test pv_mixed isa StackedPVCrosscov
        @test pv_mixed.pv_crosscovs[1] isa EvaluationPVCrosscov
        @test pv_mixed.pv_crosscovs[2] isa IntegralPVCrosscov
    end

    @testset "Application to Single PVCrosscov → Block Vector/Matrix" begin
        stacked_lf = StackedLinearFunctional([δ1, δ2])

        # Create a single PVCrosscov (randvar_arg = 2, so process is on arg 1)
        pv_single = δ3(f.kernel, arg = 2)

        # Apply stacked functional - should create column blocks
        result = stacked_lf(pv_single)
        @test result isa BlockMatrix || result isa BlockVector

        # Verify dimensions: should be (length(X1) + length(X2)) × length(X3)
        @test size(result, 1) == length(X1) + length(X2)
        @test size(result, 2) == length(X3)

        # Verify the result matches applying each functional separately
        expected_block1 = δ1(pv_single)
        expected_block2 = δ2(pv_single)
        @test result[1:length(X1), :] ≈ expected_block1
        @test result[(length(X1) + 1):end, :] ≈ expected_block2

        # Test with randvar_arg = 1 (should create row blocks)
        pv_single_arg1 = δ3(f.kernel, arg = 1)
        result_arg1 = stacked_lf(pv_single_arg1)
        @test size(result_arg1, 1) == length(X3)
        @test size(result_arg1, 2) == length(X1) + length(X2)
    end

    @testset "Application to StackedPVCrosscov → Full Block Matrix" begin
        # This is the key functionality: applying StackedLinearFunctional
        # to a StackedPVCrosscov should produce a block matrix of cross-covariances

        # 2x2 case
        stacked_lf = StackedLinearFunctional([δ1, δ2])
        pv_stack = stacked_lf(f.kernel, arg = 2)  # Creates StackedPVCrosscov

        # Apply the same stacked functional again
        block_matrix = stacked_lf(pv_stack)

        @test block_matrix isa BlockMatrix
        @test size(block_matrix) == (length(X1) + length(X2), length(X1) + length(X2))

        # Verify block structure: [[δ1∘δ1, δ1∘δ2], [δ2∘δ1, δ2∘δ2]]
        # Top-left block: δ1 applied to δ1(kernel)
        expected_11 = δ1(pv_stack.pv_crosscovs[1])
        @test block_matrix[1:length(X1), 1:length(X1)] ≈ expected_11

        # Top-right block: δ1 applied to δ2(kernel)
        expected_12 = δ1(pv_stack.pv_crosscovs[2])
        @test block_matrix[1:length(X1), (length(X1) + 1):end] ≈ expected_12

        # Bottom-left block: δ2 applied to δ1(kernel)
        expected_21 = δ2(pv_stack.pv_crosscovs[1])
        @test block_matrix[(length(X1) + 1):end, 1:length(X1)] ≈ expected_21

        # Bottom-right block: δ2 applied to δ2(kernel)
        expected_22 = δ2(pv_stack.pv_crosscovs[2])
        @test block_matrix[(length(X1) + 1):end, (length(X1) + 1):end] ≈ expected_22

        # Verify the matrix matches direct computation
        manual_matrix = [expected_11 expected_12; expected_21 expected_22]
        @test block_matrix ≈ manual_matrix

        # Test symmetry with same functionals
        stacked_lf_sym = StackedLinearFunctional([δ1, δ1])
        pv_stack_sym = stacked_lf_sym(f.kernel, arg = 2)
        block_matrix_sym = stacked_lf_sym(pv_stack_sym)

        # Should be symmetric (up to numerical precision)
        @test block_matrix_sym ≈ block_matrix_sym'
    end

    @testset "Application to StackedPVCrosscov → 3x3 Block Matrix" begin
        # Test with 3 functionals to verify generalization
        stacked_lf = StackedLinearFunctional([δ1, δ2, δ3])
        pv_stack = stacked_lf(f.kernel, arg = 2)

        block_matrix = stacked_lf(pv_stack)

        @test block_matrix isa BlockMatrix
        expected_size = length(X1) + length(X2) + length(X3)
        @test size(block_matrix) == (expected_size, expected_size)

        # Verify a few key blocks
        # Block (1,1): δ1 ∘ δ1
        expected_11 = δ1(pv_stack.pv_crosscovs[1])
        @test block_matrix[1:length(X1), 1:length(X1)] ≈ expected_11

        # Block (1,3): δ1 ∘ δ3
        expected_13 = δ1(pv_stack.pv_crosscovs[3])
        idx_start = length(X1) + length(X2) + 1
        @test block_matrix[1:length(X1), idx_start:end] ≈ expected_13

        # Block (3,2): δ3 ∘ δ2
        expected_32 = δ3(pv_stack.pv_crosscovs[2])
        @test block_matrix[idx_start:end, (length(X1) + 1):(length(X1) + length(X2))] ≈ expected_32
    end

    @testset "Application to Mean Function" begin
        stacked_lf = StackedLinearFunctional([δ1, δ2])

        # Apply to mean function
        mean_result = stacked_lf(f.mean)

        # Should stack the results vertically
        expected = vcat(δ1(f.mean), δ2(f.mean))
        @test mean_result ≈ expected
    end

    @testset "Mixed Types Application" begin
        # Test with evaluation + integral functionals
        stacked_mixed = StackedLinearFunctional([δ1, ℒ1])

        # Apply to kernel
        pv_mixed = stacked_mixed(f.kernel)
        @test pv_mixed isa StackedPVCrosscov

        # Apply to the stack to create block matrix
        block_mixed = stacked_mixed(pv_mixed)
        @test block_mixed isa BlockMatrix

        # Verify dimensions
        expected_rows = length(X1) + length(domains_1d)
        expected_cols = length(X1) + length(domains_1d)
        @test size(block_mixed) == (expected_rows, expected_cols)

        # Verify individual blocks
        expected_11 = δ1(pv_mixed.pv_crosscovs[1])  # δ1 ∘ δ1
        @test block_mixed[1:length(X1), 1:length(X1)] ≈ expected_11

        expected_12 = δ1(pv_mixed.pv_crosscovs[2])  # δ1 ∘ ℒ1
        @test block_mixed[1:length(X1), (length(X1) + 1):end] ≈ expected_12

        expected_21 = ℒ1(pv_mixed.pv_crosscovs[1])  # ℒ1 ∘ δ1
        @test block_mixed[(length(X1) + 1):end, 1:length(X1)] ≈ expected_21

        expected_22 = ℒ1(pv_mixed.pv_crosscovs[2])  # ℒ1 ∘ ℒ1
        @test block_mixed[(length(X1) + 1):end, (length(X1) + 1):end] ≈ expected_22
    end

    @testset "Edge Cases" begin
        # Single functional in stack should behave normally
        stacked_single = StackedLinearFunctional([δ1])
        pv_single_stack = stacked_single(f.kernel)

        @test pv_single_stack isa StackedPVCrosscov
        @test length(pv_single_stack.pv_crosscovs) == 1

        # Applying to itself should give a 1x1 "block" matrix
        result_single = stacked_single(pv_single_stack)
        expected_single = δ1(pv_single_stack.pv_crosscovs[1])
        @test result_single ≈ expected_single
    end

    @testset "Scaling Behavior" begin
        stacked_lf = StackedLinearFunctional([δ1, δ2])
        pv = δ3(f.kernel)

        C = 5 * rand()

        # Test scaling with single PVCrosscov
        @test stacked_lf(C * pv) ≈ C * stacked_lf(pv)

        # Test scaling with StackedPVCrosscov
        pv_stack = stacked_lf(f.kernel)
        @test stacked_lf(C * pv_stack) ≈ C * stacked_lf(pv_stack)
    end
end
