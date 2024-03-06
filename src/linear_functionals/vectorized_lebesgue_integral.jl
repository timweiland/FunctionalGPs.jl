export VectorizedLebesgueIntegral

struct VectorizedLebesgueIntegral{T <: Domain} <: AbstractLinearFunctional
    domains::AbstractVector{T}
end

function VectorizedLebesgueIntegral(domains::T...) where T <: Domain
    return VectorizedLebesgueIntegral([domains...])
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(k::CompactPolynomialKernel; arg=2) where {T}
    return CompactPolynomialCovFunc1D_Identity_LebesgueIntegral(k, ℒ.domains, arg)
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(pv::CompactPolynomialCovFunc1D_Identity_LebesgueIntegral) where {T}
    return integrate(pv.covfunc, ℒ.domains, pv.domains)
end
