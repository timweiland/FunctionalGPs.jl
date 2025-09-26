import Kronecker: kronecker

########### Kernels ###########
function (op::EvaluationFunctional)(k::Kernel; arg::Integer = 2)
    @assert arg ∈ [1, 2]
    return EvaluationPVCrosscov(k, op, arg)
end

(op::EvaluationFunctional)(k::KernelSum, args...; kwargs...) = mapreduce((k) -> op(k, args...; kwargs...), +, k.kernels)
(op::EvaluationFunctional)(k::ScaledKernel, args...; kwargs...) = k.σ² * op(k.kernel, args...; kwargs...)

########### PV Crosscovs ###########
function (op::EvaluationFunctional)(pv::EvaluationPVCrosscov{1})
    if pv.linfunc === op
        return kernelmatrix(pv.k, pv.linfunc.X)
    end
    return kernelmatrix(pv.k, pv.linfunc.X, op.X)
end

function (op::EvaluationFunctional)(pv::EvaluationPVCrosscov{2})
    if pv.linfunc === op
        return kernelmatrix(pv.k, pv.linfunc.X)
    end
    return kernelmatrix(pv.k, op.X, pv.linfunc.X)
end

function (op::EvaluationFunctional)(pv::TensorProductCrosscov)
    X = op.X

    if !(X isa FactorizedGrid)
        throw(MethodError(op, (pv,)))
    end

    factors_tuple = factors(pv)
    ndims_grid = length(X.ranges)
    if length(factors_tuple) != ndims_grid
        throw(
            DimensionMismatch(
                "Evaluation grid has $(ndims_grid) factors but tensor-product crosscovariance has $(length(factors_tuple))",
            )
        )
    end

    return mapreduce(
        args -> begin
            (i, factor) = args
            δᵢ = EvaluationFunctional(X[i])
            return δᵢ(factor)
        end,
        kronecker,
        enumerate(factors_tuple) |> collect |> reverse,
    )
end

function (ℒ::EvaluationFunctional)(
        pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral,
    )
    return kernelmatrix(pv, ℒ.X)
end
