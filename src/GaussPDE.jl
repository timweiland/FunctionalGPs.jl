module GaussPDE

export AbstractCompactKernel, AbstractCompactRadialKernel, AbstractCompactSignedRadialKernel
export CompactPolynomialKernel, CompactSignedPolynomialKernel
export derivative

export WendlandPolynomial, WendlandKernel

export Xt_A_X, Xt_A_Y, Xt_invA_X, Xt_invA_Y, diag_Xt_A_X, diag_Xt_A_Y, diag_Xt_invA_X, diag_Xt_invA_Y, tr_Xt_invA_X, Xtinv_A_Xinv

using LinearAlgebra

include("util/array_ops.jl")

include("kernels/compact.jl")
include("kernels/wendland.jl")
include("kernels/wendland_covmat_ops.jl")

include("linearfunctionoperators/linear_function_operator.jl")
end
