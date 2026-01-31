# EvaluationFunctional applied to PVCrosscovs → creates matrices

import Kronecker: kronecker

# Apply to EvaluationPVCrosscov (arg=1: first argument was evaluated)
function (op::EvaluationFunctional)(pv::EvaluationPVCrosscov{1})
    if pv.linfunc === op
        return kernel_evaluate_evaluate(pv.k, pv.linfunc.X)
    end
    return kernel_evaluate_evaluate(pv.k, pv.linfunc.X, op.X)
end

# Apply to EvaluationPVCrosscov (arg=2: second argument was evaluated)
function (op::EvaluationFunctional)(pv::EvaluationPVCrosscov{2})
    if pv.linfunc === op
        return kernel_evaluate_evaluate(pv.k, op.X)
    end
    return kernel_evaluate_evaluate(pv.k, op.X, pv.linfunc.X)
end

# Apply to TensorProductCrosscov
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

# Apply to IntegralPVCrosscov
function (ℒ::EvaluationFunctional)(pv::IntegralPVCrosscov)
    result = kernel_integrate_evaluate(pv.k, pv.domains, ℒ.X)
    return randvar_arg(pv) == 2 ? result' : result
end
