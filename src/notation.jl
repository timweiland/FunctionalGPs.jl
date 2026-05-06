"""
    FunctionalGPs.Notation

Math-flavoured aliases for the common linear functionals and operators.
Opt in with `using FunctionalGPs.Notation` so that constructions of
[`FunctionalGaussian`](@ref) read like the math:

```julia
using FunctionalGPs, FunctionalGPs.Notation

fg = FunctionalGaussian(f;
    y  = δ(X_obs),
    dy = δ(X_pred) ∘ ∂(1),
    q  = ∫([Interval(0, 1)]),
)
```

Symbols exported:

- `δ` — point-evaluation functional ([`EvaluationFunctional`](@ref)).
- `∂` — first-order partial derivative in 1D ([`PartialDerivative`](@ref)).
  Higher-order or higher-dimensional partials use `PartialDerivative(multi_idx)`
  directly: `PartialDerivative((1, 0))` for ∂/∂x₁ in 2D,
  `PartialDerivative((1, 1))` for the mixed ∂²/∂x₁∂x₂, etc.
- `∫` — vectorised Lebesgue integral ([`VectorizedLebesgueIntegral`](@ref)).

These are package-local aliases — they do not collide with `Differential` from
Symbolics/ModelingToolkit.
"""
module Notation

import ..FunctionalGPs:
    EvaluationFunctional, PartialDerivative, VectorizedLebesgueIntegral

export δ, ∂, ∫

"""
    δ(X::AbstractVector)
    δ(x::Real)

Point-evaluation (Dirac) functional. Equivalent to
[`EvaluationFunctional`](@ref).
"""
δ(X::AbstractVector) = EvaluationFunctional(X)
δ(x::Real) = EvaluationFunctional([x])

"""
    ∂(i::Integer)

First-order partial derivative in 1D — ∂/∂xᵢ. Equivalent to
`PartialDerivative((i,))`.

For higher-order or higher-dimensional partial derivatives, use the multi-index
form `PartialDerivative((i₁, i₂, ...))` directly. Note that
`∂(1) ∘ ∂(1)` is *not* automatically folded into `PartialDerivative((2,))`;
it stays as a composition. Use `PartialDerivative((2,))` for ∂²/∂x₁².
"""
∂(i::Integer) = PartialDerivative((i,))

"""
    ∫(args...)

Vectorised Lebesgue integral functional. Forwards to
[`VectorizedLebesgueIntegral`](@ref); accepts the same argument forms (a
domain, a vararg list of domains, or an array of domains).
"""
∫(args...) = VectorizedLebesgueIntegral(args...)

end
