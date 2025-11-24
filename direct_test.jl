using Pkg
Pkg.activate(".")

# Add test dependencies
Pkg.add("ReTest")

# Load the package
using GaussPDE

# Load test dependencies
using AbstractGPs
using KernelFunctions
import GaussPDE: randvar_batch_size, randvar_arg, randproc_arg
import KernelFunctions: kernelmatrix
import LinearAlgebra: eigvals

# Load ReTest for test macros
using ReTest

# Include and run the test file
include("test/linear_functionals/arithmetic/test_stack.jl")

# Run all tests
retest()
