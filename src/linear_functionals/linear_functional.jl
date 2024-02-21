import AbstractGPs: ZeroMean

export AbstractLinearFunctional, output_shape

abstract type AbstractLinearFunctional end
output_shape(op::AbstractLinearFunctional) = op.output_shape

(ℒ::AbstractLinearFunctional)(::ZeroMean{T}, args...) where {T} = zeros(T, output_shape(ℒ)...)

function (ℒ::AbstractLinearFunctional)(pv::EvaluationPVCrosscov)
    return kernelmatrix(ℒ(pv.k, arg=randproc_arg(pv)), pv.X)
end

function (ℒ::AbstractLinearFunctional)(pv::StackedPVCrosscov)
    blocks = Tuple(map(ℒ, pv.pv_crosscovs))
    if randproc_arg(pv) == 1
        return mortar(blocks)
    else
        return mortar(Tuple(tuple(block) for block in blocks)...)
    end
end

function (ℒ::AbstractLinearFunctional)(pv::AbstractSumPVCrosscov)
    return sum([ℒ(pv_crosscov) for pv_crosscov in pv.summands])
end