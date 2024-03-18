module GaussPDE

using LinearAlgebra

include("util/array_ops.jl")
include("util/unified_cholesky.jl")

include("factorized_grid.jl")

include("domains/domain.jl")
include("domains/interval.jl")
include("domains/box.jl")
include("domains/factorized_box.jl")

include("kernels/compact.jl")
include("kernels/radial.jl")
include("kernels/wendland.jl")

include("pv_crosscovs/pv_crosscov.jl")
include("pv_crosscovs/evaluation.jl")
include("pv_crosscovs/stack.jl")
include("pv_crosscovs/arithmetic/scale.jl")
include("pv_crosscovs/arithmetic/sum.jl")
include("pv_crosscovs/integrals/radial.jl")
include("pv_crosscovs/integrals/compact_polynomial.jl")

include("linearfunctionoperators/linear_function_operator.jl")
include("linearfunctionoperators/arithmetic/concatenate.jl")
include("linearfunctionoperators/arithmetic/sum.jl")
include("linearfunctionoperators/arithmetic/scale.jl")
include("linearfunctionoperators/differential_operators/differential_operator.jl")
include("linearfunctionoperators/differential_operators/partial_derivative.jl")
include("linearfunctionoperators/differential_operators/linear_diffop.jl")

include("linear_functionals/linear_functional.jl")
include("linear_functionals/evaluation.jl")
include("linear_functionals/arithmetic/concatenate.jl")
include("linear_functionals/arithmetic/sum.jl")
include("linear_functionals/vectorized_lebesgue_integral.jl")

include("randprocs/linfctl_transformed_gp.jl")
include("randprocs/linear_conditional_gp.jl")

include("problems/ibvp.jl")
include("problems/heat.jl")

end
