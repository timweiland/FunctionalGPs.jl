export AbstractLinearFunctionOperator

abstract type AbstractLinearFunctionOperator end

# Arithmetic operations on operators (needed by differential operators)
include("sum.jl")
include("scale.jl")
include("concatenate.jl")

# Differential operators
include("differential.jl")
include("partial_derivative.jl")
include("linear_diffop.jl")
