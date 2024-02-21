module GaussPDETests

using GaussPDE
using ReTest
using Aqua
import SparseArrays.CHOLMOD.Factor as SparseCholFactor
using Distributions: MvNormal

include("util/test_array_ops.jl")
include("util/test_unified_cholesky.jl")

include("test_factorized_grid.jl")

include("kernels/test_wendland.jl")

include("linearfunctionoperators/arithmetic/test_concatenate.jl")
include("linearfunctionoperators/differential_operators/test_partial_derivative.jl")
include("linearfunctionoperators/differential_operators/test_pd_wendland.jl")
include("linearfunctionoperators/test_evaluation.jl")

@testset "Aqua" begin
    Aqua.test_all(GaussPDE; piracies = false, ambiguities = false)
    Aqua.test_piracies(GaussPDE; treat_as_own = [MvNormal, SparseCholFactor])
    @test length(Test.detect_ambiguities(GaussPDE)) == 0
end

end