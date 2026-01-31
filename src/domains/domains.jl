export Domain

abstract type Domain end

volume(d::Domain) = throw(MethodError(volume, (d,)))
Base.in(_, d::Domain) = throw(MethodError(in, (d,)))

# Concrete domain types
include("interval.jl")
include("box.jl")
include("factorized_box.jl")
include("grids.jl")
