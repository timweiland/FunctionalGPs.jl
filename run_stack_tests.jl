using Pkg
Pkg.activate(".")

# Run only the StackedLinearFunctional tests and capture all output
println("Running StackedLinearFunctional tests...")
println("="^80)

try
    Pkg.test("GaussPDE"; test_args=["linear_functionals/arithmetic/test_stack"])
    println("\n" * "="^80)
    println("✓ ALL STACKED LINEAR FUNCTIONAL TESTS PASSED!")
catch e
    println("\n" * "="^80)
    println("✗ TESTS FAILED")
    println("\nError details:")
    println(sprint(showerror, e))
end
