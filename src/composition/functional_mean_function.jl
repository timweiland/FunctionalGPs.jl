import AbstractGPs: mean_vector

function _functional_mean_fn_fallback end

(ℒ::AbstractLinearFunctional)(m::MeanFunction, args...; kwargs...) = _functional_mean_fn_fallback(ℒ, m, args...; kwargs...)

_functional_mean_fn_fallback(ℒ::EvaluationFunctional, m, args...; kwargs...) = mean_vector(m, ℒ.X)

_functional_mean_fn_fallback(ℒ::AbstractLinFctlLinFuncOpConcat, args...; kwargs...) = _concat_crosscov_impl(ℒ, args...; kwargs...)
_functional_mean_fn_fallback(ℒ::AbstractSumLinearFunctional, args...; kwargs...) = _sum_crosscov_impl(ℒ, args...; kwargs...)
_functional_mean_fn_fallback(ℒ::ScaledLinearFunctional, args...; kwargs...) = _scale_crosscov_impl(ℒ, args...; kwargs...)

_functional_mean_fn_fallback(ℒ::StackedLinearFunctional, m, args...; kwargs...) = begin
    # Flatten each component before stacking: components may return arrays of
    # differing shape (e.g. a flat `Fill` from a `ConstMean` vs. a multi-axis
    # `zeros(output_shape...)` from a `ZeroMean`), which `vcat` cannot combine
    # unless reduced to vectors first.
    results = [vec(lf(m, args...; kwargs...)) for lf in ℒ.linfunctionals]
    return mortar(results)
end

# Any linear functional applied to the zero function is zero.
(ℒ::AbstractLinearFunctional)(::ZeroMean{T}, args...; kwargs...) where {T} = zeros(T, output_shape(ℒ)...)
