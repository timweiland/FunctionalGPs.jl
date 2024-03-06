module GaussPDETests

using GaussPDE
using ReTest
using Aqua
import SparseArrays.CHOLMOD.Factor as SparseCholFactor
using Distributions: MvNormal

include("util/test_array_ops.jl")
include("util/test_unified_cholesky.jl")

include("test_factorized_grid.jl")

include("domains/test_box.jl")
include("domains/test_interval.jl")

include("kernels/test_wendland.jl")

include("pv_crosscovs/arithmetic/test_scale.jl")
include("pv_crosscovs/integrals/test_wendland.jl")

include("linearfunctionoperators/arithmetic/test_concatenate.jl")
include("linearfunctionoperators/arithmetic/test_scale.jl")
include("linearfunctionoperators/arithmetic/test_sum.jl")
include("linearfunctionoperators/differential_operators/test_partial_derivative.jl")
include("linearfunctionoperators/differential_operators/test_pd_wendland.jl")
include("linearfunctionoperators/differential_operators/test_linear_diffop.jl")
include("linearfunctionoperators/test_evaluation.jl")

include("linear_functionals/test_linear_functional.jl")
include("linear_functionals/arithmetic/test_sum.jl")
include("linear_functionals/arithmetic/test_concatenate.jl")

include("test_sin_regression.jl")
include("problems/test_heat.jl")

@testset "Aqua" begin
    Aqua.test_all(GaussPDE; piracies = false, ambiguities = false)
    Aqua.test_piracies(GaussPDE; treat_as_own = [MvNormal, SparseCholFactor])
    @test length(Test.detect_ambiguities(GaussPDE)) == 0
end

end