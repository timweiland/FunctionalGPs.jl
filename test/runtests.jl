using GaussPDE
using Test
using Aqua
import SparseArrays.CHOLMOD.Factor as SparseCholFactor

@testset "GaussPDE.jl" begin
    include("util/test_array_ops.jl")
    include("test_factorized_grid.jl")

    include("kernels/test_wendland.jl")

    include("linearfunctionoperators/arithmetic/test_concatenate.jl")

    Aqua.test_all(GaussPDE; piracies=false, ambiguities=false)
    Aqua.test_piracies(GaussPDE; treat_as_own=[SparseCholFactor])
    @test length(Test.detect_ambiguities(GaussPDE)) == 0
end
