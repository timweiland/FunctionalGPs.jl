export VectorizedLebesgueIntegral

struct VectorizedLebesgueIntegral{T <: Domain} <: AbstractLinearFunctional
    domains::AbstractVector{T}

    function VectorizedLebesgueIntegral(domains::AbstractVector{T}) where T
        if length(domains) == 0
            throw(ArgumentError("At least one domain must be provided"))
        end
        return new{T}(domains)
    end

    function VectorizedLebesgueIntegral(domains...)
        return VectorizedLebesgueIntegral(domains)
    end
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(k::CompactPolynomialKernel; arg=2) where {T}
    return CompactPolynomialCovFunc1D_Identity_LebesgueIntegral(k, ℒ.domains, arg)
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(pv::CompactPolynomialCovFunc1D_Identity_LebesgueIntegral) where {T}
    return integrate(pv.covfunc, ℒ.domains, pv.domains)
end
