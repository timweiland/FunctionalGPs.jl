export EvaluationFunctional

"""
    EvaluationFunctional <: AbstractLinearFunctional

Point evaluation functional that evaluates a function at specified input locations.

When applied to a GP kernel, this produces the standard covariance matrix between
evaluation points. This is the most common functional used in GP regression.

# Fields
- `X::AbstractVector`: Input locations where the function is evaluated
- `output_shape::Tuple`: Shape of the output (automatically set to `size(X)`)

# Example
```julia
# Evaluate at three points
X = [0.0, 0.5, 1.0]
δ = EvaluationFunctional(X)

# Compose with differentiation to get derivative observations
∂x = PartialDerivative((1,))
δ_dx = δ ∘ ∂x  # Evaluates the derivative at X
```

# See also
- [`VectorizedLebesgueIntegral`](@ref): For integral observations
- [`StackedLinearFunctional`](@ref): To combine with other functionals
"""
struct EvaluationFunctional <: AbstractLinearFunctional
    X::AbstractVector
    output_shape::Tuple{Vararg{Integer}}
end

"""
    EvaluationFunctional(X::AbstractVector)

Construct an evaluation functional at the input locations `X`.
"""
function EvaluationFunctional(X::AbstractVector)
    return EvaluationFunctional(X, size(X))
end
