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
    blocks = [f(pv_component) for f in stacked.functionals, pv_component in pv.pv_crosscovs]

    # Convert to block matrix
    # The orientation depends on which argument the PV crosscov was applied to
    if randproc_arg(pv) == 1
        # When applied to arg=1, we're building columns
        # Each row in 'blocks' becomes a row in the block matrix
        return mortar(Tuple(Tuple(blocks[i, :]) for i in 1:size(blocks, 1))...)
    else
        # When applied to arg=2, we're building rows
        # Each column in 'blocks' becomes a column in the block matrix
        return mortar(Tuple(Tuple(blocks[:, j]) for j in 1:size(blocks, 2))...)
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
function (stacked::StackedLinearFunctional)(k::KernelSum, args...; kwargs...)
    # Linearity: distribute over sum
    return mapreduce((kern) -> stacked(kern, args...; kwargs...), +, k.kernels)
end

function (stacked::StackedLinearFunctional)(k::ScaledKernel, args...; kwargs...)
    # Linearity: factor out scalar
    return k.σ² * stacked(k.kernel, args...; kwargs...)
end
