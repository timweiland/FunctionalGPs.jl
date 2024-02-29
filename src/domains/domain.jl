export Domain

abstract type Domain end

volume(d::Domain) = error("volume not implemented for $(typeof(d))")
Base.in(::AbstractVector, d::Domain) = error("in not implemented for $(typeof(d))")
