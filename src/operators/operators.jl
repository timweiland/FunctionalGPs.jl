export AbstractLinearFunctionOperator

"""
    AbstractLinearFunctionOperator

Abstract supertype for linear operators that act on functions (e.g., differentiation).

Operators can be composed with functionals using `∘` to create new functionals.
For example, `EvaluationFunctional(X) ∘ PartialDerivative((1,))` creates a functional
that evaluates the first derivative at points `X`.

# Subtypes
- [`PartialDerivative`](@ref): Partial differentiation operator
- `LinearDifferentialOperator`: Linear combinations of partial derivatives
"""
abstract type AbstractLinearFunctionOperator end

# Arithmetic operations on operators (needed by differential operators)
include("sum.jl")
include("scale.jl")
include("concatenate.jl")

# Identity (zeroth-order term in operator algebra)
include("identity.jl")

# Select (multi-output: pick an output index)
include("select.jl")

# Differential operators
include("differential.jl")
include("partial_derivative.jl")
include("linear_diffop.jl")
