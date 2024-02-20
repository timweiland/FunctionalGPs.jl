using ReTest, GaussPDE
include("GaussPDETests.jl")

if "skip-aqua" in ARGS
    GaussPDETests.retest(r"\b(?!Aqua\b)\w+")
else
    GaussPDETests.retest()
end
