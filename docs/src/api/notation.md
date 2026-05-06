# Notation

The `FunctionalGPs.Notation` submodule provides math-flavoured aliases for
the common linear functionals and operators. It is opt-in — load it
explicitly when you want construction of a [`FunctionalGaussian`](@ref) to
read like the math:

```julia
using FunctionalGPs, FunctionalGPs.Notation

fg = FunctionalGaussian(f;
    y  = δ(X_obs),
    dy = δ(X_pred) ∘ ∂(1),
    q  = ∫([Interval(0.0, 1.0)]),
)
```

| Symbol | Equivalent to | Meaning |
|--------|---------------|---------|
| `δ(X)` | `EvaluationFunctional(X)` | Point evaluation (Dirac) |
| `δ(x::Real)` | `EvaluationFunctional([x])` | Single-point evaluation |
| `∂(i::Integer)` | `PartialDerivative((i,))` | First-order partial in 1D — ∂/∂xᵢ |
| `∫(args...)` | `VectorizedLebesgueIntegral(args...)` | Lebesgue integral over one or more domains |

## Higher-order and higher-dimensional partials

`∂` is intentionally restricted to a single integer argument because
[`PartialDerivative`](@ref) uses a multi-index where the tuple position
encodes the dimension and the value encodes the per-dimension order. A single
`∂(i)` matches only the unambiguous 1D case. For higher orders or higher
dimensions, use the multi-index form directly:

```julia
PartialDerivative((2,))     # ∂²/∂x₁² in 1D
PartialDerivative((1, 0))   # ∂/∂x₁ in 2D
PartialDerivative((1, 1))   # mixed ∂²/∂x₁∂x₂ in 2D
```

Composition does *not* fold: `∂(1) ∘ ∂(1)` builds a
`ConcatenatedLinearFunctionOperator`, not `PartialDerivative((2,))`. They are
equivalent in value but not in type. Prefer the multi-index form when you
mean a higher-order partial.

## Conflicts

`∂` is exported by neither `Symbolics` nor `ModelingToolkit`, so loading
those packages alongside `FunctionalGPs.Notation` does not produce a clash.
`Differential` from those packages is deliberately *not* aliased here.
