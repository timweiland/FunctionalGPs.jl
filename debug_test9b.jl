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

println("Testing individual functional applications:")
println("δ1(pv_single) size: ", size(δ1(pv_single)))
println("δ2(pv_single) size: ", size(δ2(pv_single)))

println("\nExpected:")
println("  δ1(pv_single): (3, 3) - 3 output points from δ1, 3 input points from pv")
println("  δ2(pv_single): (2, 3) - 2 output points from δ2, 3 input points from pv")
println("  Stacked: (5, 3) - total 5 output points, 3 input points")
