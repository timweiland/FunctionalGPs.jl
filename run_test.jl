using TestEnv

TestEnv.activate()

include("test/GaussPDETests.jl")

GaussPDETests.retest(ARGS[1])
