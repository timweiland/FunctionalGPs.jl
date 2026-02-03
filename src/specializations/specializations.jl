# Specializations: trait-dispatched implementations for specific kernel types
#
# These provide optimized implementations of kernel operations:
# - Stationary kernels → Toeplitz matrices, closed-form integrals
# - Compact kernels → Sparse matrices via spatial search
# - Matern kernels → Closed-form derivatives and integrals

include("stationary/stationary.jl")
include("compact/compact.jl")
include("matern/matern.jl")
include("squared_exponential/squared_exponential.jl")
