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

println("pv_single type: ", typeof(pv_single))
println("pv_single randvar_batch_size: ", GaussPDE.randvar_batch_size(pv_single))

result = stacked(pv_single)

println("\nResult type: ", typeof(result))
println("Result size: ", size(result))
println("Expected size: (", length(X1) + length(X2), ", ", length(X1), ")")

println("\nResult is a BlockMatrix with these properties:")
println("  Number of block rows: ", length(result.blocks))
println("  Number of block cols in first row: ", length(result.blocks[1]))

for i in 1:length(result.blocks)
    for j in 1:length(result.blocks[i])
        println("  Block[$i, $j] size: ", size(result.blocks[i][j]))
    end
end
