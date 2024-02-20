import AbstractGPs: ZeroMean

export AbstractLinearFunctional

abstract type AbstractLinearFunctional end
output_shape(op::AbstractLinearFunctional) = op.output_shape

(ℒ::AbstractLinearFunctional)(::ZeroMean{T}, args...) where {T} = zeros(T, output_shape(ℒ)...)

function (ℒ::AbstractLinearFunctional)(pv::EvaluationPVCrosscov)
    return kernelmatrix(ℒ(pv.k, arg=randproc_arg(pv)), pv.X)
end
