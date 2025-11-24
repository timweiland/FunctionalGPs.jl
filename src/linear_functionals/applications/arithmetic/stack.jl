using BlockArrays

"""
    (stacked::StackedLinearFunctional)(k::Kernel; arg=2)

Apply a stacked linear functional to a kernel, creating a StackedPVCrosscov where each
component is the result of applying each individual functional to the kernel.

# Arguments
- `k::Kernel`: The kernel to apply to
- `arg::Int=2`: Which argument of the kernel to apply to (1 or 2)

# Returns
- `StackedPVCrosscov`: A stacked process-vector crosscovariance
"""
function (stacked::StackedLinearFunctional)(k::Kernel; arg::Integer = 2)
    pv_crosscovs = [lf(k, arg = arg) for lf in stacked.linfunctionals]
    return StackedPVCrosscov(pv_crosscovs)
end

"""
    _apply_to_single_pv(stacked::StackedLinearFunctional, pv)

Internal helper to apply stacked functional to a single process-vector crosscovariance,
creating a block vector or block matrix depending on randproc_arg.
"""
function _apply_to_single_pv(stacked::StackedLinearFunctional, pv)
    blocks = [lf(pv) for lf in stacked.linfunctionals]
    n_blocks = length(blocks)

    # Stack the blocks appropriately based on randproc_arg
    # If randproc_arg == 1, we're stacking along the first dimension (column blocks - nx1)
    # If randproc_arg == 2, we're stacking along the second dimension (row blocks - 1xn)
    if randproc_arg(pv) == 1
        # Process is on first argument, so we create column blocks (stack vertically)
        return mortar(reshape(blocks, n_blocks, 1))
    else
        # Process is on second argument, so we create row blocks (stack horizontally)
        return mortar(reshape(blocks, 1, n_blocks))
    end
end

"""
    (stacked::StackedLinearFunctional)(pv::ProcessVectorCrossCovariance)

Apply a stacked linear functional to a single process-vector crosscovariance, creating
a block vector or block matrix depending on the randvar_arg of the input.

# Arguments
- `pv::ProcessVectorCrossCovariance`: The process-vector crosscovariance to apply to

# Returns
- Block array (vector or matrix) with results stacked appropriately
"""
function (stacked::StackedLinearFunctional)(pv::ProcessVectorCrossCovariance)
    return _apply_to_single_pv(stacked, pv)
end

"""
    (stacked::StackedLinearFunctional)(pv::EvaluationPVCrosscov)

Specific method to resolve ambiguity with AbstractLinearFunctional.
"""
function (stacked::StackedLinearFunctional)(pv::EvaluationPVCrosscov)
    return _apply_to_single_pv(stacked, pv)
end

"""
    (stacked::StackedLinearFunctional)(pv::ConstantScaledPVCrosscov)

Specific method to resolve ambiguity with AbstractLinearFunctional.
Distributes the scale factor across the stacked functionals.
"""
function (stacked::StackedLinearFunctional)(pv::ConstantScaledPVCrosscov)
    return scale(pv) * stacked(pv.pv_crosscov)
end

"""
    (stacked::StackedLinearFunctional)(pv_stack::StackedPVCrosscov)

Apply a stacked linear functional to a StackedPVCrosscov, creating a full block matrix
where each block (i,j) is the result of applying functional i to the j-th component
of the StackedPVCrosscov.

# Arguments
- `pv_stack::StackedPVCrosscov`: The stacked process-vector crosscovariance

# Returns
- `BlockMatrix`: A block matrix of cross-covariances
"""
function (stacked::StackedLinearFunctional)(pv_stack::StackedPVCrosscov)
    n_functionals = length(stacked.linfunctionals)
    n_pv_crosscovs = length(pv_stack.pv_crosscovs)

    # Create vector of blocks in row-major order: [block[1,1], block[1,2], ..., block[2,1], ...]
    blocks = [
        stacked.linfunctionals[i](pv_stack.pv_crosscovs[j])
            for i in 1:n_functionals, j in 1:n_pv_crosscovs
    ]

    # Convert to BlockMatrix using mortar with proper shape
    return mortar(blocks)
end

"""
    (stacked::StackedLinearFunctional)(f::Function)

Apply a stacked linear functional to a mean function, stacking the results vertically.

# Arguments
- `f::Function`: The mean function to apply to

# Returns
- Vertically stacked results of applying each functional to the mean function
"""
function (stacked::StackedLinearFunctional)(f::Function)
    results = [lf(f) for lf in stacked.linfunctionals]
    return vcat(results...)
end
