using KernelFunctions: Kernel

export EvaluationPVCrosscov

struct EvaluationPVCrosscov <: ProcessVectorCrossCovariance
    k::Kernel
    X::AbstractVector
    eval_arg::Integer

    function EvaluationPVCrosscov(k::Kernel, X::AbstractVector, eval_arg::Integer)
        @assert eval_arg ∈ [1, 2]
        return new(k, X, eval_arg)
    end
end

randvar_batch_size(pv::EvaluationPVCrosscov) = size(pv.X)
randvar_arg(pv::EvaluationPVCrosscov) = pv.eval_arg
