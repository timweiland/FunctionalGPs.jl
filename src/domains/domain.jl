export Domain

abstract type Domain end

volume(d::Domain) = error("volume not implemented for $(typeof(d))")
Base.in(_, d::Domain) = error("in not implemented for $(typeof(d))")
