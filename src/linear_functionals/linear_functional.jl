export AbstractLinearFunctional, output_shape

abstract type AbstractLinearFunctional end
output_shape(op::AbstractLinearFunctional) = op.output_shape
