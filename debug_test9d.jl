using Pkg
Pkg.activate(".")

using GaussPDE
using AbstractGPs
using KernelFunctions

k = SqExponentialKernel()
X1 = [0.0, 0.5, 1.0]
X2 = [0.25, 0.75]
δ1 = EvaluationFunctional(X1)
δ2 = EvaluationFunctional(X2)

stacked = StackedLinearFunctional([δ1, δ2])
pv_single = δ1(k; arg=1)

println("Checking method dispatch:")
println("pv_single type: ", typeof(pv_single))
println()

# Check which methods are available
println("Available methods for StackedLinearFunctional applied to EvaluationPVCrosscov:")
println(methods(stacked, (typeof(pv_single),)))

println("\nCalling stacked(pv_single)...")
result = stacked(pv_single)
println("Result type: ", typeof(result))
println("Result size: ", size(result))
