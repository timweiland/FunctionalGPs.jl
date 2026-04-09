using KernelFunctions: Kernel, KernelTensorProduct

export EvaluationPVCrosscov

"""
    EvaluationPVCrosscov{EvalArg, TK, TL}

PV crosscov representing point evaluation applied to one argument of a kernel.
This is the generic type used for evaluation functionals.

The `EvalArg` type parameter (1 or 2) indicates which kernel argument the
evaluation is applied to.

# Fields
- `eval_arg::Int`: Which argument (1 or 2) the evaluation is applied to
- `k::TK`: The kernel
- `linfunc::TL`: The evaluation functional containing the points

# Examples
```julia
julia> k = HalfIntegerMaternKernel(2, [1.0]);
julia> X = [0.0, 0.5, 1.0];
julia> ℒ = EvaluationFunctional(X);
julia> pv = ℒ(k, arg=1);  # Evaluate first argument
julia> typeof(pv)
EvaluationPVCrosscov{1, ...}
```

# See also
- [`IntegralPVCrosscov`](@ref): For integration functionals
- [`kernel_evaluate_evaluate`](@ref): For building covariance matrices
"""
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
    kernel_evaluate_evaluate(pv.k, pv.linfunc.X, X)
kernelmatrix(pv::EvaluationPVCrosscov{2}, X::AbstractVector) =
    kernel_evaluate_evaluate(pv.k, X, pv.linfunc.X)

# KernelTensorProduct with multi-dimensional points: route through
# kernel_evaluate_evaluate for lazy per-dimension dispatch.
# Depending on input types this produces:
# - FactorizedGrid × FactorizedGrid → Kronecker product
# - FactorizedGrid × vector-of-vectors → KhatriRao product
# - vector-of-vectors × vector-of-vectors → Hadamard product
function kernelmatrix(
        pv::EvaluationPVCrosscov{1, <:KernelTensorProduct},
        X::AbstractVector{<:AbstractVector},
    )
    return kernel_evaluate_evaluate(pv.k, pv.linfunc.X, X)
end

function kernelmatrix(
        pv::EvaluationPVCrosscov{2, <:KernelTensorProduct},
        X::AbstractVector{<:AbstractVector},
    )
    return kernel_evaluate_evaluate(pv.k, X, pv.linfunc.X)
end

function Base.isequal(pv1::EvaluationPVCrosscov, pv2::EvaluationPVCrosscov)
    return pv1.k == pv2.k && pv1.linfunc == pv2.linfunc && pv1.eval_arg == pv2.eval_arg
end

function Base.isapprox(pv1::EvaluationPVCrosscov, pv2::EvaluationPVCrosscov)
    return pv1.k ≈ pv2.k && pv1.linfunc.X ≈ pv2.linfunc.X && pv1.eval_arg == pv2.eval_arg
end
