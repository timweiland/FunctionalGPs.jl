using KernelFunctions: Kernel

export EvaluationPVCrosscov

struct EvaluationPVCrosscov{EvalArg, TK <: Kernel, TL} <: ProcessVectorCrossCovariance
    eval_arg::Int
    k::TK
    linfunc::TL
end

function EvaluationPVCrosscov(k::Kernel, linfunc::EvaluationFunctional, eval_arg::Integer)
    arg = Int(eval_arg)
    @assert arg ∈ (1, 2)
    return EvaluationPVCrosscov{arg, typeof(k), typeof(linfunc)}(arg, k, linfunc)
end

randvar_batch_size(pv::EvaluationPVCrosscov) = size(pv.linfunc.X)
randvar_arg(::EvaluationPVCrosscov{EvalArg}) where {EvalArg} = EvalArg

kernelmatrix(pv::EvaluationPVCrosscov{1}, X::AbstractVector) =
    kernelmatrix(pv.k, pv.linfunc.X, X)
kernelmatrix(pv::EvaluationPVCrosscov{2}, X::AbstractVector) =
    kernelmatrix(pv.k, X, pv.linfunc.X)

function Base.isequal(pv1::EvaluationPVCrosscov, pv2::EvaluationPVCrosscov)
    return pv1.k == pv2.k && pv1.linfunc == pv2.linfunc && pv1.eval_arg == pv2.eval_arg
end

function Base.isapprox(pv1::EvaluationPVCrosscov, pv2::EvaluationPVCrosscov)
    return pv1.k ≈ pv2.k && pv1.linfunc.X ≈ pv2.linfunc.X && pv1.eval_arg == pv2.eval_arg
end
