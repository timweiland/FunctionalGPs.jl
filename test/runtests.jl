using GaussPDE
using Test
using Aqua
import SparseArrays.CHOLMOD.Factor as SparseCholFactor

@testset "GaussPDE.jl" begin
    include("kernels/test_wendland.jl")

    Aqua.test_all(GaussPDE; piracies=false, ambiguities=false)
    Aqua.test_piracies(GaussPDE; treat_as_own=[SparseCholFactor])
    @test length(Test.detect_ambiguities(GaussPDE)) == 0
end
