# Generic AbstractLinearFunctional applied to Kernels

# Any linear functional applied to the zero kernel produces the zero crosscov.
function (ℒ::AbstractLinearFunctional)(::ZeroKernel; arg::Integer = 2)
    argi = Int(arg)
    @assert argi ∈ (1, 2)
    return ZeroPVCrosscov(output_shape(ℒ), argi)
end

# Handle KernelSum - apply to each component
function (ℒ::AbstractLinearFunctional)(k::KernelSum, args...; kwargs...)
    return mapreduce((k) -> ℒ(k, args...; kwargs...), +, k.kernels)
end

# Handle ScaledKernel - scale the result
function (ℒ::AbstractLinearFunctional)(k::ScaledKernel, args...; kwargs...)
    return k.σ² * ℒ(k.kernel, args...; kwargs...)
end

# Handle LinearlyScaledKernel - same pattern, allows negative scalars
function (ℒ::AbstractLinearFunctional)(k::LinearlyScaledKernel, args...; kwargs...)
    return k.scalar * ℒ(k.kernel, args...; kwargs...)
end

# Compose a functional with the operator stored on the pinned argument of a
# TransformedMultiOutputKernel. The identity operator drops out so the common
# (pure-`Select`) case stays a bare functional.
_compose_functional(ℒ::AbstractLinearFunctional, ::Identity) = ℒ
_compose_functional(ℒ::AbstractLinearFunctional, op) = ℒ ∘ op

# Apply any linear functional to a half-pinned TransformedMultiOutputKernel. The
# functional is first composed with the operator the pinned argument carries, so a
# deferred derivative/scaling acts on the eventual single-output block. Hitting the
# already-pinned argument (`arg == Arg`) leaves the other output free (→
# MultiOutputPVCrosscov holding `functional ∘ op`); hitting the other argument
# determines both outputs, so the free one is pinned with `Select`.
#
# This is a plain helper rather than something the concrete functionals reach via
# `invoke`: `EvaluationFunctional` / `VectorizedLebesgueIntegral` carry their own
# generic `(::F)(::Kernel)` methods (ambiguous with the `AbstractLinearFunctional`
# method below), and `invoke` can only generalise positional argument types, never
# the receiver/functional type — so their disambiguating methods forward here.
function _functional_on_transformed(
        ℒ::AbstractLinearFunctional,
        tmk::TransformedMultiOutputKernel{K, Arg};
        arg = 2,
    ) where {K <: MultiOutputKernel, Arg}
    combined = _compose_functional(ℒ, spatial_op(tmk))
    p = pinned_output(tmk)
    return (arg == Arg) ?
        MultiOutputPVCrosscov{Arg}(tmk.parent, p, combined) :
        Select(p)(combined(tmk.parent; arg = arg))
end

(ℒ::AbstractLinearFunctional)(tmk::TransformedMultiOutputKernel{<:MultiOutputKernel}; arg = 2) =
    _functional_on_transformed(ℒ, tmk; arg = arg)
