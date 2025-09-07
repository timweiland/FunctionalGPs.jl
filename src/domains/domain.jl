export Domain

abstract type Domain end

volume(d::Domain) = throw(MethodError(volume, (d,)))
Base.in(_, d::Domain) = throw(MethodError(in, (d,)))
