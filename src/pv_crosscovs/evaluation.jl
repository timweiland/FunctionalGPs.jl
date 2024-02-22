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

function kernelmatrix(pv::EvaluationPVCrosscov, X::AbstractVector)
    if pv.eval_arg == 1
        return kernelmatrix(pv.k, pv.X, X)
    else
        return kernelmatrix(pv.k, X, pv.X)
    end
end

function Base.isequal(pv1::EvaluationPVCrosscov, pv2::EvaluationPVCrosscov)
    return pv1.k == pv2.k && pv1.X == pv2.X && pv1.eval_arg == pv2.eval_arg
end

function Base.isapprox(pv1::EvaluationPVCrosscov, pv2::EvaluationPVCrosscov)
    return pv1.k ≈ pv2.k && pv1.X ≈ pv2.X && pv1.eval_arg == pv2.eval_arg
end