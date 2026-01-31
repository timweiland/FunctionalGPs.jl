export AbstractLinearFunctional, output_shape

abstract type AbstractLinearFunctional end
output_shape(op::AbstractLinearFunctional) = op.output_shape

# Concrete functional types
include("evaluation.jl")
include("integral.jl")

# Arithmetic operations on functionals
include("sum.jl")
include("concatenate.jl")
include("stacked.jl")
include("tensor_product.jl")
