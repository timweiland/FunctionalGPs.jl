using Pkg
Pkg.activate(".")

using GaussPDE
using AbstractGPs
using KernelFunctions
import GaussPDE: randvar_batch_size, randvar_arg, randproc_arg

k = SqExponentialKernel()
k_integrable = WendlandKernel(1, 3)
X1 = [0.0, 0.5, 1.0]
domains1 = [Interval(0.0, 1.0), Interval(1.0, 2.0)]

δ1 = EvaluationFunctional(X1)
ℒ1 = VectorizedLebesgueIntegral(domains1)

# Test what randproc_arg returns for different PVCrosscovs
pv1 = δ1(k; arg=1)
pv2 = δ1(k; arg=2)
pv3 = ℒ1(k_integrable; arg=1)

println("δ1(k; arg=1):")
println("  randproc_arg: ", randproc_arg(pv1))
println("  randvar_arg: ", randvar_arg(pv1))
println("  randvar_batch_size: ", randvar_batch_size(pv1))

println("\nδ1(k; arg=2):")
println("  randproc_arg: ", randproc_arg(pv2))
println("  randvar_arg: ", randvar_arg(pv2))
println("  randvar_batch_size: ", randvar_batch_size(pv2))

println("\nℒ1(k_integrable; arg=1):")
println("  randproc_arg: ", randproc_arg(pv3))
println("  randvar_arg: ", randvar_arg(pv3))
println("  randvar_batch_size: ", randvar_batch_size(pv3))

# Now test what happens when we apply functionals
println("\n\nApplying δ1 to pv3:")
result1 = δ1(pv3)
println("  Size: ", size(result1))
println("  Expected: (3, 2) [output_shape × randvar_batch_size]")

println("\nApplying ℒ1 to pv1:")
result2 = ℒ1(pv1)
println("  Size: ", size(result2))
println("  Expected: (2, 3) [output_shape × randvar_batch_size]")

# Test stacked
println("\n\nTesting StackedLinearFunctional:")
stacked = StackedLinearFunctional([δ1, ℒ1])
pv_stacked = stacked(k_integrable; arg=1)
println("Stacked PV randproc_arg: ", randproc_arg(pv_stacked))

println("\nApplying functionals to stacked PV components:")
blocks_11 = δ1(pv_stacked.pv_crosscovs[1])
blocks_12 = δ1(pv_stacked.pv_crosscovs[2])
blocks_21 = ℒ1(pv_stacked.pv_crosscovs[1])
blocks_22 = ℒ1(pv_stacked.pv_crosscovs[2])

println("  blocks[1,1] size: ", size(blocks_11), " (expected (3,3))")
println("  blocks[1,2] size: ", size(blocks_12), " (expected (3,2))")
println("  blocks[2,1] size: ", size(blocks_21), " (expected (2,3))")
println("  blocks[2,2] size: ", size(blocks_22), " (expected (2,2))")
