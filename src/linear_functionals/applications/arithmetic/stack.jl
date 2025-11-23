using AbstractGPs
using BlockArrays

"""
    (stacked::StackedLinearFunctional)(k::Kernel; arg=2)

Apply a stacked linear functional to a kernel, producing a `StackedPVCrosscov`.

Each functional in the stack is applied to the kernel, and the resulting
process-vector cross-covariances are stacked together.

# Arguments
- `k::Kernel`: The kernel to apply the functionals to
- `arg::Integer=2`: Which argument of the kernel to apply to (1 or 2)

# Returns
- `StackedPVCrosscov`: A stacked cross-covariance with one component per functional
"""
function (stacked::StackedLinearFunctional)(k::Kernel; arg::Integer = 2)
    pv_crosscovs = [f(k; arg=arg) for f in stacked.functionals]
    return StackedPVCrosscov(pv_crosscovs)
end

"""
    (stacked::StackedLinearFunctional)(pv::StackedPVCrosscov)

Apply a stacked linear functional to a stacked PV cross-covariance, producing a block matrix.

This is the key operation for creating symmetric block matrices when applying
the stacked functional from both sides of a kernel.

# Arguments
- `pv::StackedPVCrosscov`: The stacked cross-covariance to apply to

# Returns
- A block matrix (via `mortar`) where block (i,j) corresponds to applying
  functional i to the kernel and functional j to the other side.
"""
function (stacked::StackedLinearFunctional)(pv::StackedPVCrosscov)
    # Apply each functional to each component of the stacked PV crosscov
    # This creates a matrix of blocks
    blocks_raw = [f(pv_component) for f in stacked.functionals, pv_component in pv.pv_crosscovs]

    # Ensure each block has the correct orientation
    # Block (i,j) should have rows corresponding to functional i's output size
    # and columns corresponding to the pv_component j's batch size
    blocks = Matrix{AbstractMatrix}(undef, size(blocks_raw))
    for i in 1:size(blocks_raw, 1)
        row_size = prod(output_shape(stacked[i]))
        for j in 1:size(blocks_raw, 2)
            # The column size comes from the pv component's randvar batch size
            col_size = prod(randvar_batch_size(pv.pv_crosscovs[j]))
            current_size = size(blocks_raw[i, j])

            # Check if we need to transpose to get the right orientation
            if current_size == (row_size, col_size)
                blocks[i, j] = blocks_raw[i, j]
            elseif current_size == (col_size, row_size)
                blocks[i, j] = blocks_raw[i, j]'
            else
                error("Unexpected block size at position ($i, $j): got $current_size, expected ($row_size, $col_size) or ($col_size, $row_size)")
            end
        end
    end

    # Convert to block matrix
    # Each row in 'blocks' becomes a row in the block matrix
    return mortar(Tuple(Tuple(blocks[i, :]) for i in 1:size(blocks, 1))...)
end

"""
    (stacked::StackedLinearFunctional)(::ZeroMean{T})

Apply a stacked linear functional to a zero mean function.

# Returns
- A vector of zero arrays, one for each functional's output shape
"""
function (stacked::StackedLinearFunctional)(m::ZeroMean{T}) where {T}
    # Apply each functional to the zero mean
    results = [f(m) for f in stacked.functionals]
    # Stack the results vertically
    return vcat(results...)
end

# Support for other PV crosscov types
function (stacked::StackedLinearFunctional)(pv::EvaluationPVCrosscov)
    # Apply to kernel from the other side, then kernelmatrix
    return kernelmatrix(stacked(pv.k, arg = randproc_arg(pv)), pv.linfunc.X)
end

function (stacked::StackedLinearFunctional)(pv::AbstractSumPVCrosscov)
    # Linearity: distribute over sum
    return sum([stacked(pv_component) for pv_component in pv.summands])
end

function (stacked::StackedLinearFunctional)(pv::ConstantScaledPVCrosscov)
    # Linearity: factor out scalar
    return scale(pv) * stacked(pv.pv_crosscov)
end

# Support for kernel types
function (stacked::StackedLinearFunctional)(k::KernelSum; arg::Integer = 2)
    # Linearity: distribute over sum
    return mapreduce((kern) -> stacked(kern; arg=arg), +, k.kernels)
end

function (stacked::StackedLinearFunctional)(k::ScaledKernel; arg::Integer = 2)
    # Apply to the underlying kernel - each individual functional will see the ScaledKernel
    # and handle it appropriately, ensuring the scalar is applied at the right level
    pv_crosscovs = [f(k; arg=arg) for f in stacked.functionals]
    return StackedPVCrosscov(pv_crosscovs)
end
