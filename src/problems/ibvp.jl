export IBVP, domain, lindiffops, sample_ic, sample_bc, solution

abstract type IBVP end
domain(p::IBVP) = error("domain not defined for $(typeof(p))")
lindiffops(p::IBVP) = error("lindiffops not defined for $(typeof(p))")
sample_ic(p::IBVP, args...; kwargs...) = error("sample_ic not defined for $(typeof(p))")
sample_bc(p::IBVP, args...; kwargs...) = error("sample_bc not defined for $(typeof(p))")
solution(p::IBVP) = error("solution not defined for $(typeof(p))")
