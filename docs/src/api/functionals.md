# Functionals

Linear functionals are the core abstraction in FunctionalGPs.jl. A linear functional maps a function to a finite-dimensional vector. The most common examples are:

- **Point evaluation**: Evaluate a function at specific input locations
- **Integration**: Integrate a function over specified domains
- **Derivative evaluation**: Evaluate derivatives at specific locations (via composition with differential operators)

## Creating Functionals

### Point Evaluation

```julia
using FunctionalGPs

# Evaluate at three points
X = [0.0, 0.5, 1.0]
δ = EvaluationFunctional(X)
```

### Integration

```julia
using FunctionalGPs
using DomainSets

# Integrate over an interval
∫ = VectorizedLebesgueIntegral(Interval(0, 1))

# Integrate over multiple intervals
intervals = [Interval(0, 1), Interval(1, 2)]
∫_vec = VectorizedLebesgueIntegral(intervals)
```

### Derivative Evaluation

Compose an evaluation functional with a differential operator:

```julia
using FunctionalGPs

δ = EvaluationFunctional([0.0, 0.5, 1.0])
∂x = PartialDerivative((1,))
δ_dx = δ ∘ ∂x  # Evaluate the first derivative
```

## Combining Functionals

Functionals can be combined using three operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Sum of functionals (same output shape required) | `δ1 + δ2` |
| `∘` | Composition with differential operators | `δ ∘ ∂x` |
| `⊗` | Tensor product for separable domains | `δ_x ⊗ ∫_y` |

### Stacking Functionals

Use [`StackedLinearFunctional`](@ref) to combine functionals with different output shapes:

```julia
δ = EvaluationFunctional(X)
∂x = PartialDerivative((1,))
δ_dx = δ ∘ ∂x

# Stack function values and derivative observations
stacked = StackedLinearFunctional(δ, δ_dx)
```

### Tensor Products

Use `⊗` for multi-dimensional domains with separable structure:

```julia
# Evaluate at x, integrate over y (axis-aligned line integrals)
k = k_x ⊗ k_y  # Tensor product kernel
δ_x = EvaluationFunctional(x_points)
∫_y = VectorizedLebesgueIntegral(y_intervals)
ℒ = δ_x ⊗ ∫_y
```

## API Reference

### Point Evaluation

```@docs
EvaluationFunctional
```

### Integration

```@docs
VectorizedLebesgueIntegral
```

### Differential Operators

```@docs
PartialDerivative
AbstractLinearFunctionOperator
```

### Combining Functionals

```@docs
StackedLinearFunctional
TensorProductFunctional
SumLinearFunctional
LinFctlLinFuncOpConcat
```

### Abstract Types

```@docs
AbstractLinearFunctional
AbstractSumLinearFunctional
AbstractLinFctlLinFuncOpConcat
```
