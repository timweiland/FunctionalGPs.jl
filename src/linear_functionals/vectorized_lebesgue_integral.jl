export VectorizedLebesgueIntegral

struct VectorizedLebesgueIntegral{T <: Domain} <: AbstractLinearFunctional
    domains::AbstractArray{T}

    function VectorizedLebesgueIntegral(domains::AbstractArray{T}) where {T}
        if Base.length(domains) == 0
            throw(ArgumentError("At least one domain must be provided"))
        end
        return new{T}(domains)
    end

    VectorizedLebesgueIntegral(domains...) = VectorizedLebesgueIntegral(domains)
end

output_shape(ℒ::VectorizedLebesgueIntegral) = size(ℒ.domains)
