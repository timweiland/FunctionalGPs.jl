using Pkg
Pkg.activate(".")

using GaussPDE
using AbstractGPs
using KernelFunctions
using BlockArrays
import GaussPDE: output_shape, randvar_batch_size

k = SqExponentialKernel()
X1 = [0.0, 0.5, 1.0]
X2 = [0.25, 0.75]
δ1 = EvaluationFunctional(X1)
δ2 = EvaluationFunctional(X2)

stacked = StackedLinearFunctional([δ1, δ2])
pv_single = δ1(k; arg=1)

println("Manual simulation of the generic method:")
println("="^60)

# Apply each functional
blocks_raw = [f(pv_single) for f in stacked.functionals]
println("blocks_raw sizes:")
for i in 1:length(blocks_raw)
    println("  blocks_raw[$i]: ", size(blocks_raw[i]))
end

# Transpose correction
blocks = Vector{AbstractMatrix}(undef, length(blocks_raw))
for i in 1:length(blocks_raw)
    row_size = prod(output_shape(stacked[i]))
    col_size = prod(randvar_batch_size(pv_single))
    current_size = size(blocks_raw[i])

    println("\nBlock $i:")
    println("  row_size (output_shape): $row_size")
    println("  col_size (randvar_batch_size): $col_size")
    println("  current_size: $current_size")

    if current_size == (row_size, col_size)
        blocks[i] = blocks_raw[i]
        println("  -> No transpose needed")
    elseif current_size == (col_size, row_size)
        blocks[i] = blocks_raw[i]'
        println("  -> Transposing to ", size(blocks[i]))
    end
end

println("\nblocks (after transpose) sizes:")
for i in 1:length(blocks)
    println("  blocks[$i]: ", size(blocks[i]))
end

# Create block matrix
println("\nCreating block matrix...")
result = mortar(Tuple(Tuple([block]) for block in blocks)...)
println("Result type: ", typeof(result))
println("Result size: ", size(result))
