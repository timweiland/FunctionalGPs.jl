# Squared Exponential kernel specializations
#
# Provides optimized implementations for:
# - Stationary kernel specs (lazy matrix construction)
# - Radial antiderivatives (closed-form integration)

include("stationary_spec.jl")
include("integration.jl")
