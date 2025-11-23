using Pkg
Pkg.activate(".")

# Run tests
result = try
    Pkg.test("GaussPDE")
    println("\n\n✓ ALL TESTS PASSED!")
    0
catch e
    println("\n\n✗ TESTS FAILED")
    println(sprint(showerror, e))
    1
end

exit(result)
