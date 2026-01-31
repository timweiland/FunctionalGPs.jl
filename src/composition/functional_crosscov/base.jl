# Generic AbstractLinearFunctional applied to PVCrosscovs → creates matrices

import AbstractGPs: ZeroMean

# Mean functions (needed for GP conditioning)
(ℒ::AbstractLinearFunctional)(::ZeroMean{T}, args...) where {T} = zeros(T, output_shape(ℒ)...)

# Apply to EvaluationPVCrosscov
function (ℒ::AbstractLinearFunctional)(pv::EvaluationPVCrosscov)
    return kernelmatrix(ℒ(pv.k, arg = randproc_arg(pv)), pv.linfunc.X)
end

# Apply to StackedPVCrosscov
function (ℒ::AbstractLinearFunctional)(pv::StackedPVCrosscov)
    blocks = Tuple(map(ℒ, pv.pv_crosscovs))
    if randproc_arg(pv) == 1
        return mortar(blocks)
    else
        return mortar(Tuple(tuple(block) for block in blocks)...)
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
