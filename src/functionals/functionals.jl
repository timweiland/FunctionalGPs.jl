export AbstractLinearFunctional, output_shape

"""
    AbstractLinearFunctional

Abstract supertype for all linear functionals on functions.

A linear functional is a linear map from a function space to a finite-dimensional vector space.
Common examples include point evaluation, integration over domains, and compositions with
differential operators.

Linear functionals can be combined using:
- `+` : Sum of functionals (must have same output shape)
- `∘` : Composition with a [`AbstractLinearFunctionOperator`](@ref) (e.g., differentiation)
- `⊗` : Tensor product for multi-dimensional domains

# Implementing a new functional

Custom functionals should subtype `AbstractLinearFunctional` and implement:
- `output_shape(ℒ)`: Return a tuple specifying the output dimensions

# See also
- [`EvaluationFunctional`](@ref): Point evaluation
- [`VectorizedLebesgueIntegral`](@ref): Integration over domains
- [`StackedLinearFunctional`](@ref): Stack multiple functionals
- [`TensorProductFunctional`](@ref): Tensor product of functionals
"""
abstract type AbstractLinearFunctional end

"""
    output_shape(ℒ::AbstractLinearFunctional) -> Tuple

Return the output shape of the functional as a tuple of integers.

The output shape determines the dimensions of the resulting matrix when the functional
is applied to a kernel from both sides.
"""
output_shape(op::AbstractLinearFunctional) = op.output_shape

# Concrete functional types
include("evaluation.jl")
include("integral.jl")

# Arithmetic operations on functionals
include("scale.jl")
include("sum.jl")
include("concatenate.jl")
include("stacked.jl")
include("tensor_product.jl")
