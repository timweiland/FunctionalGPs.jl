using Pkg
Pkg.activate(".")

using GaussPDE
using AbstractGPs
using KernelFunctions
import GaussPDE: randvar_batch_size, randvar_arg, randproc_arg
import KernelFunctions: kernelmatrix
import LinearAlgebra: eigvals

println("="^80)
println("Manual StackedLinearFunctional Tests")
println("="^80)

# Setup
k = SqExponentialKernel()
k_integrable = WendlandKernel(1, 3)

X1 = [0.0, 0.5, 1.0]
X2 = [0.25, 0.75]
δ1 = EvaluationFunctional(X1)
δ2 = EvaluationFunctional(X2)

domains1 = [Interval(0.0, 1.0), Interval(1.0, 2.0)]
domains2 = [Interval(0.5, 1.5)]
ℒ1 = VectorizedLebesgueIntegral(domains1)
ℒ2 = VectorizedLebesgueIntegral(domains2)

tests_passed = 0
tests_failed = 0

function run_test(name::String, test_func::Function)
    global tests_passed, tests_failed
    print("Testing: $name ... ")
    try
        test_func()
        println("✓ PASS")
        tests_passed += 1
    catch e
        println("✗ FAIL")
        println("  Error: ", sprint(showerror, e))
        tests_failed += 1
    end
end

# Test 1: Constructor
run_test("Constructor with vector", () -> begin
    stacked = StackedLinearFunctional([δ1, δ2])
    @assert length(stacked) == 2
end)

# Test 2: Varargs constructor
run_test("Constructor with varargs", () -> begin
    stacked = StackedLinearFunctional(δ1, δ2)
    @assert length(stacked) == 2
end)

# Test 3: Apply to kernel arg=2
run_test("Apply to kernel (arg=2)", () -> begin
    stacked = StackedLinearFunctional([δ1, δ2])
    pv = stacked(k; arg=2)
    @assert pv isa StackedPVCrosscov
    @assert length(pv.pv_crosscovs) == 2
end)

# Test 4: Apply to kernel arg=1
run_test("Apply to kernel (arg=1)", () -> begin
    stacked = StackedLinearFunctional([δ1, δ2])
    pv = stacked(k; arg=1)
    @assert pv isa StackedPVCrosscov
    @assert randvar_arg(pv) == 1
end)

# Test 5: Symmetric block matrix (evaluation only)
run_test("Symmetric block matrix: Evaluation only", () -> begin
    stacked = StackedLinearFunctional([δ1, δ2])
    pv1 = stacked(k; arg=1)
    result = stacked(pv1)

    # Verify dimensions
    @assert size(result, 1) == length(X1) + length(X2)
    @assert size(result, 2) == length(X1) + length(X2)

    # Verify symmetry
    @assert result ≈ result'
end)

# Test 6: Mixed functionals (evaluation + integral)
run_test("Mixed functionals (evaluation + integral)", () -> begin
    stacked = StackedLinearFunctional([δ1, ℒ1])
    pv1 = stacked(k_integrable; arg=1)
    result = stacked(pv1)

    # Verify dimensions
    @assert size(result, 1) == length(X1) + length(domains1)
    @assert size(result, 2) == length(X1) + length(domains1)

    # Verify symmetry
    @assert result ≈ result'
end)

# Test 7: Asymmetric application
run_test("Asymmetric application (different functionals)", () -> begin
    stacked1 = StackedLinearFunctional([δ1, ℒ1])
    stacked2 = StackedLinearFunctional([δ2, ℒ2])

    pv1 = stacked1(k_integrable; arg=1)
    result = stacked2(pv1)

    # Verify dimensions (should be non-square)
    @assert size(result, 1) == length(X2) + length(domains2)
    @assert size(result, 2) == length(X1) + length(domains1)
end)

# Test 8: Non-square block matrix (arg=2)
run_test("Non-square block matrix (arg=2)", () -> begin
    stacked1 = StackedLinearFunctional([δ1, ℒ1])
    stacked2 = StackedLinearFunctional([δ2, ℒ2])

    pv2 = stacked1(k_integrable; arg=2)
    result = stacked2(pv2)

    # Should have different dimensions than the arg=1 case
    @assert size(result, 1) == length(X2) + length(domains2)
    @assert size(result, 2) == length(X1) + length(domains1)
end)

# Test 9: Apply to non-stacked PV (EvaluationPVCrosscov)
run_test("Apply to single EvaluationPVCrosscov", () -> begin
    stacked = StackedLinearFunctional([δ1, δ2])
    pv_single = δ1(k; arg=1)
    result = stacked(pv_single)

    # Should create a column vector of results (one per functional)
    @assert size(result, 1) == length(X1) + length(X2)
    @assert size(result, 2) == length(X1)
end)

# Test 10: Apply to non-stacked PV (IntegralPVCrosscov)
run_test("Apply to single IntegralPVCrosscov", () -> begin
    stacked = StackedLinearFunctional([δ1, δ2])
    pv_single = ℒ1(k_integrable; arg=1)
    result = stacked(pv_single)

    # Should create a column vector of results
    @assert size(result, 1) == length(X1) + length(X2)
    @assert size(result, 2) == length(domains1)
end)

println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("Tests passed: $tests_passed")
println("Tests failed: $tests_failed")
println("="^80)

if tests_failed > 0
    exit(1)
end
