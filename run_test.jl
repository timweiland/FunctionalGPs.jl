using TestEnv

TestEnv.activate()

include("test/FunctionalGPsTests.jl")

FunctionalGPsTests.retest(ARGS[1])
