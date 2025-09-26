import AbstractGPs: ZeroMean

########## Mean functions ##########
(ℒ::AbstractLinearFunctional)(::ZeroMean{T}, args...) where {T} = zeros(T, output_shape(ℒ)...)

########## PV Crosscovs ##########
function (ℒ::AbstractLinearFunctional)(pv::EvaluationPVCrosscov)
    return kernelmatrix(ℒ(pv.k, arg = randproc_arg(pv)), pv.linfunc.X)
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

function (ℒ::AbstractLinearFunctional)(pv::ConstantScaledPVCrosscov)
    return scale(pv) * ℒ(pv.pv_crosscov)
end

########## Kernels ##########
function (ℒ::AbstractLinearFunctional)(k::KernelSum, args...; kwargs...)
    return mapreduce((k) -> ℒ(k, args...; kwargs...), +, k.kernels)
end

function (ℒ::AbstractLinearFunctional)(k::ScaledKernel, args...; kwargs...)
    return k.σ² * ℒ(k.kernel, args...; kwargs...)
end
