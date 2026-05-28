module FunctionalGPsTests

using FunctionalGPs
using ReTest
using Aqua
import SparseArrays.CHOLMOD.Factor as SparseCholFactor
import Kronecker: CholeskyKronecker
using Distributions: MvNormal

include("test_utils/quad.jl")

# === Utilities ===
include("util/test_array_ops.jl")
include("util/test_cholesky.jl")

# === Domains ===
include("domains/test_grids.jl")
include("domains/test_box.jl")
include("domains/test_interval.jl")
include("domains/test_factorized_box.jl")

# === Kernels ===
include("kernels/test_wendland.jl")
include("kernels/test_compact_kernelmatrix.jl")
include("kernels/test_matern_derivative.jl")
include("kernels/test_tensor_product_traits.jl")
include("kernels/test_squared_exponential.jl")

# === Matrices ===
include("matrices/test_khatri_rao.jl")

# === Crosscov ===
include("crosscov/test_scale.jl")
include("crosscov/test_tensor_product.jl")
include("crosscov/test_wendland_integrals.jl")
include("crosscov/test_radial_integrals.jl")
include("crosscov/test_se_integrals.jl")

# === Operators ===
include("operators/test_concatenate.jl")
include("operators/test_scale.jl")
include("operators/test_sum.jl")
include("operators/test_partial_derivative.jl")
include("operators/test_pd_wendland.jl")
include("operators/test_linear_diffop.jl")
include("operators/test_evaluation.jl")
include("operators/test_identity.jl")

# === Functionals ===
include("functionals/test_linear_functional.jl")
include("functionals/test_scale.jl")
include("functionals/test_sum.jl")
include("functionals/test_concatenate.jl")
include("functionals/test_stack.jl")
include("functionals/test_tensor_product.jl")
include("functionals/test_vectorized_lebesgue_integral.jl")
include("functionals/test_integral_derivative_interplay.jl")
include("functionals/test_factorized_integrals.jl")

# === GPs ===
include("gps/test_functional_gaussian.jl")

# === Notation submodule ===
include("test_notation.jl")

# === Vecchia / GMRF bridge ===
include("test_vecchia.jl")

# === Integration tests ===
include("test_sin_regression.jl")
include("problems/test_heat.jl")

# === Autodiff ===
include("hyperopt/test_autodiff.jl")

@testset "Aqua" begin
    Aqua.test_all(FunctionalGPs; piracies = false, ambiguities = false)
    Aqua.test_piracies(FunctionalGPs; treat_as_own = [MvNormal, SparseCholFactor, CholeskyKronecker])
    # TODO: Fix 38 method ambiguities (mostly CholeskyKronecker \ ArrayLayouts conflicts)
    # @test length(Test.detect_ambiguities(FunctionalGPs)) == 0
end

end
