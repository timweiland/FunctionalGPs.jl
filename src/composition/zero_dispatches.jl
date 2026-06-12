# Completes the zero short-circuit dispatch.
#
# The generic short-circuits
#   (::AbstractLinearFunctional)(::ZeroKernel)
#   (::AbstractLinearFunctional)(::ZeroPVCrosscov)
#   (::AbstractLinearFunctionOperator)(::ZeroPVCrosscov)
# are ambiguous with the per-type methods of the *composite* functionals and
# operators (sum / scaled / concatenation / stacked functionals; scaled /
# concatenation / sum / identity operators), each of which has a generic
# `(op)(x, args...)` method. `EvaluationFunctional` and `VectorizedLebesgueIntegral`
# were already disambiguated for `ZeroKernel`; these methods cover the rest.
#
# Every method preserves the absorbing behaviour: a functional applied to a zero
# kernel is the zero crosscov; to a zero crosscov is a zero matrix; an operator
# applied to a zero crosscov leaves it unchanged.

function _zero_pv_matrix(ℒ, pv::ZeroPVCrosscov)
    n = randvar_length(pv)
    m = prod(output_shape(ℒ))
    return pv.randvar_arg == 1 ? zeros(n, m) : zeros(m, n)
end

# Composite functionals applied to the zero kernel → zero crosscov.
(ℒ::AbstractSumLinearFunctional)(::ZeroKernel; arg::Integer = 2) =
    ZeroPVCrosscov(output_shape(ℒ), arg)
(ℒ::ScaledLinearFunctional)(::ZeroKernel; arg::Integer = 2) =
    ZeroPVCrosscov(output_shape(ℒ), arg)
(ℒ::AbstractLinFctlLinFuncOpConcat)(::ZeroKernel; arg::Integer = 2) =
    ZeroPVCrosscov(output_shape(ℒ), arg)
(ℒ::StackedLinearFunctional)(::ZeroKernel; arg::Integer = 2) =
    ZeroPVCrosscov(output_shape(ℒ), arg)

# Composite functionals applied to the zero crosscov → zero matrix.
(ℒ::AbstractSumLinearFunctional)(pv::ZeroPVCrosscov) = _zero_pv_matrix(ℒ, pv)
(ℒ::ScaledLinearFunctional)(pv::ZeroPVCrosscov) = _zero_pv_matrix(ℒ, pv)
(ℒ::AbstractLinFctlLinFuncOpConcat)(pv::ZeroPVCrosscov) = _zero_pv_matrix(ℒ, pv)
(ℒ::StackedLinearFunctional)(pv::ZeroPVCrosscov) = _zero_pv_matrix(ℒ, pv)

# Composite operators applied to the zero crosscov → unchanged (absorbing).
(op::AbstractScaledLinearFunctionOperator)(pv::ZeroPVCrosscov) = pv
(op::AbstractConcatenatedLinearFunctionOperator)(pv::ZeroPVCrosscov) = pv
(op::AbstractSumLinearFunctionOperator)(pv::ZeroPVCrosscov) = pv
(op::Identity)(pv::ZeroPVCrosscov) = pv
