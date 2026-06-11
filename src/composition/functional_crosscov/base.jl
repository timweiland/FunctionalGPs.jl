# Generic AbstractLinearFunctional applied to PVCrosscovs → creates matrices

# Apply to EvaluationPVCrosscov
function (ℒ::AbstractLinearFunctional)(pv::EvaluationPVCrosscov)
    return kernelmatrix(ℒ(pv.k, arg = randproc_arg(pv)), pv.linfunc.X)
end

# Apply to StackedPVCrosscov
function (ℒ::AbstractLinearFunctional)(pv::StackedPVCrosscov)
    blocks = collect(map(ℒ, pv.pv_crosscovs))
    if randproc_arg(pv) == 1
        # Process on arg 1: blocks have shape (n_integrals × n_randvars)
        # randvars vary between blocks (different column counts)
        # Stack horizontally (1×n block row)
        return mortar(reshape(blocks, 1, length(blocks)))
    else
        # Process on arg 2: blocks have shape (n_randvars × n_integrals)
        # randvars vary between blocks (different row counts)
        # Stack vertically (n×1 block column)
        return mortar(reshape(blocks, length(blocks), 1))
    end
end

# Apply to AbstractSumPVCrosscov
function (ℒ::AbstractLinearFunctional)(pv::AbstractSumPVCrosscov)
    return sum([ℒ(pv_crosscov) for pv_crosscov in pv.summands])
end

# Apply to ConstantScaledPVCrosscov
function (ℒ::AbstractLinearFunctional)(pv::ConstantScaledPVCrosscov)
    return scale(pv) * ℒ(pv.pv_crosscov)
end
