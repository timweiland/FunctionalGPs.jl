using ReTest, FunctionalGPs
include("FunctionalGPsTests.jl")

if "skip-aqua" in ARGS
    FunctionalGPsTests.retest(r"\b(?!Aqua\b)\w+")
else
    FunctionalGPsTests.retest()
end
