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

    # Transpose blocks if needed based on the arg attribute
    # The orientation depends on which argument the PV crosscov was applied to
    if randproc_arg(pv) == 1
        # Building row-wise: blocks in same row must have same number of rows
        # Each block should have rows = functional output size
        blocks = Matrix{AbstractMatrix}(undef, size(blocks_raw))
        for i in 1:size(blocks_raw, 1)
            expected_rows = prod(output_shape(stacked[i]))
            for j in 1:size(blocks_raw, 2)
                current_size = size(blocks_raw[i, j])
                # If rows don't match expected, try transpose
                if current_size[1] != expected_rows && current_size[2] == expected_rows
                    blocks[i, j] = blocks_raw[i, j]'
                else
                    blocks[i, j] = blocks_raw[i, j]
                end
            end
        end
        return mortar(Tuple(Tuple(blocks[i, :]) for i in 1:size(blocks, 1))...)
    else
        # Building column-wise: blocks in same column must have same number of columns
        # Each block should have columns = pv component batch size
        blocks = Matrix{AbstractMatrix}(undef, size(blocks_raw))
        for j in 1:size(blocks_raw, 2)
            expected_cols = prod(randvar_batch_size(pv.pv_crosscovs[j]))
            for i in 1:size(blocks_raw, 1)
                current_size = size(blocks_raw[i, j])
                # If cols don't match expected, try transpose
                if current_size[2] != expected_cols && current_size[1] == expected_cols
                    blocks[i, j] = blocks_raw[i, j]'
                else
                    blocks[i, j] = blocks_raw[i, j]
                end
            end
        end
        # mortar takes rows, so always use row-wise construction
        return mortar(Tuple(Tuple(blocks[i, :]) for i in 1:size(blocks, 1))...)
    end
end

"""
    (stacked::StackedLinearFunctional)(pv::ProcessVectorCrossCovariance)

Apply a stacked linear functional to a single (non-stacked) ProcessVectorCrossCovariance.
This creates a block vector where each block corresponds to applying one of the functionals
to the given PV.

# Arguments
- `pv::ProcessVectorCrossCovariance`: A single PV cross-covariance

# Returns
- A block matrix with one column, where row i contains the result of applying functional i to pv
"""
function (stacked::StackedLinearFunctional)(pv::ProcessVectorCrossCovariance)
    # Apply each functional to the single PV crosscov
    # This creates a vector of blocks
    # Note: We transpose each block because functional application returns transposed results
    blocks = [f(pv)' for f in stacked.functionals]

    # Convert to block matrix
    # The orientation depends on which argument the PV crosscov was applied to
    if randproc_arg(pv) == 1
        # When randproc_arg==1, we're building rows
        # All blocks go in a single row
        return mortar((tuple(blocks...),)...)
    else
        # When randproc_arg==2, we're building columns
        # Each block becomes a row in the block matrix
        return mortar(Tuple(Tuple([block]) for block in blocks)...)
    end
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

# Specific methods to resolve ambiguity with AbstractLinearFunctional methods
function (stacked::StackedLinearFunctional)(pv::EvaluationPVCrosscov)
    # Delegate to the generic ProcessVectorCrossCovariance method
    return invoke(stacked, Tuple{ProcessVectorCrossCovariance}, pv)
end

function (stacked::StackedLinearFunctional)(pv::IntegralPVCrosscov)
    # Delegate to the generic ProcessVectorCrossCovariance method
    return invoke(stacked, Tuple{ProcessVectorCrossCovariance}, pv)
end

# Support for other PV crosscov types

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
