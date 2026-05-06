module FunctionalGPs

using LinearAlgebra

# === Utilities ===
include("util/array_ops.jl")
include("util/cholesky.jl")
include("util/kronecker.jl")

# === Layer 1: Building blocks ===
include("domains/domains.jl")
include("kernels/kernels.jl")
include("operators/operators.jl")
include("functionals/functionals.jl")

# === Layer 2: Intermediate representation ===
include("crosscov/crosscov.jl")

# === Layer 3: Output matrices ===
include("matrices/matrices.jl")

# === Specializations (trait-dispatched implementations) ===
include("specializations/specializations.jl")

# === Composition (how layers connect) ===
include("composition/composition.jl")

# === High-level APIs ===
include("gps/gps.jl")
include("problems/problems.jl")

# === Vecchia approximation interface ===
# Categorisation/coordinates/policy live here; the GMRF-returning method for
# `vecchia(::FunctionalGaussian)` is in ext/FunctionalGPsGMRFsExt.jl.
include("vecchia/main.jl")

# === Notation submodule (opt-in math-style aliases) ===
include("notation.jl")

# include("hyperopt/hyperopt.jl")  # Still disabled

# === AD support ===
include("chainrules.jl")

end
