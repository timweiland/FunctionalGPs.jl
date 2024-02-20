module GaussPDE

using LinearAlgebra

include("util/array_ops.jl")

include("kernels/compact.jl")
include("kernels/wendland.jl")
include("kernels/wendland_covmat_ops.jl")

include("pv_crosscovs/pv_crosscov.jl")
include("pv_crosscovs/evaluation.jl")

include("linearfunctionoperators/linear_function_operator.jl")
include("linearfunctionoperators/arithmetic/concatenate.jl")
include("linearfunctionoperators/arithmetic/sum.jl")
include("linearfunctionoperators/differential_operators/differential_operator.jl")
include("linearfunctionoperators/differential_operators/partial_derivative.jl")
include("linearfunctionoperators/differential_operators/linear_diffop.jl")

include("linear_functionals/linear_functional.jl")
include("linear_functionals/evaluation.jl")

include("factorized_grid.jl")

end
