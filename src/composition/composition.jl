# Composition: how functionals and operators transform kernels and crosscovs
#
# This module contains the dispatch logic that connects:
# - Functionals + Kernels → Crosscovs
# - Functionals + Crosscovs → Matrices
# - Operators + Kernels → DerivativeKernels

using AbstractGPs: MeanFunction, ZeroMean
using KernelFunctions: ZeroKernel

include("functional_kernel/functional_kernel.jl")
include("functional_crosscov/functional_crosscov.jl")
include("functional_mean_function.jl")
include("operator_kernel/operator_kernel.jl")
include("zero_dispatches.jl")
