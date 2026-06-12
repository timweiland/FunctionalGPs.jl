import KernelFunctions: KernelSum, ScaledKernel

######### Mean functions #########
(::AbstractLinearFunctionOperator)(::ZeroMean{T}, args...) where {T} = ZeroMean{T}()

######### PV Crosscovs #########
(::AbstractLinearFunctionOperator)(pv::ZeroPVCrosscov) = pv

function (ℒ::AbstractLinearFunctionOperator)(pv::EvaluationPVCrosscov)
    return EvaluationPVCrosscov(ℒ(pv.k; arg = randproc_arg(pv)), pv.linfunc, randvar_arg(pv))
end

function (ℒ::AbstractLinearFunctionOperator)(pv::StackedPVCrosscov)
    return StackedPVCrosscov(map(ℒ, pv.pv_crosscovs))
end

function (ℒ::AbstractLinearFunctionOperator)(pv::AbstractSumPVCrosscov)
    return SumPVCrosscov(map(ℒ, pv.summands))
end

function (ℒ::AbstractLinearFunctionOperator)(pv::ConstantScaledPVCrosscov)
    return ConstantScaledPVCrosscov(ℒ(pv.pv_crosscov), pv.scalar)
end

######### Kernels #########
function (ℒ::AbstractLinearFunctionOperator)(k::KernelSum, args...; kwargs...)
    return mapreduce((k) -> ℒ(k, args...; kwargs...), +, k.kernels)
end

function (ℒ::AbstractLinearFunctionOperator)(k::ScaledKernel, args...; kwargs...)
    return ScaledKernel(ℒ(k.kernel, args...; kwargs...), k.σ²)
end

function (ℒ::AbstractLinearFunctionOperator)(k::LinearlyScaledKernel, args...; kwargs...)
    return LinearlyScaledKernel(ℒ(k.kernel, args...; kwargs...), k.scalar)
end
