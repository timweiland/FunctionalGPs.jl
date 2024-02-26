export LinearDifferentialOperator,
    PartialDerivativeCoefficients,
    MultiIndex,
    MultiDict,
    PartialDerivativeCoefficients,
    MaybeScaledPartialDerivative

const MultiIndex{M} = NTuple{M,Int}
const MultiDict{M,T} = Dict{MultiIndex{M},T} where {T<:Real}
const PartialDerivativeCoefficients{M,T} = Dict{Int,D} where {M,T<:Real,D<:MultiDict{M,T}}
const MaybeScaledPartialDerivative{N,M} = Union{
    ConstantScaledLinearFunctionOperator{PartialDerivative{N,M}},
    PartialDerivative{N,M},
}

struct LinearDifferentialOperator{K,N,M} <: AbstractSumLinearFunctionOperator{K}
    idx_dict::PartialDerivativeCoefficients{M}
    summands::NTuple{K,MaybeScaledPartialDerivative{N,M}}

    function LinearDifferentialOperator{N}(
        idx_dict::PartialDerivativeCoefficients{M},
    ) where {N,M}
        if !all(1 <= key <= N for key in keys(idx_dict))
            throw(DomainError("Keys of idx_dict must be between 1 and $N"))
        end
        summands = Vector{MaybeScaledPartialDerivative{N,M}}()
        for (output_idx, multi_idx_dict) in idx_dict
            for (multi_idx, coeff) in multi_idx_dict
                push!(summands, coeff * PartialDerivative{N,M}(output_idx, multi_idx))
            end
        end
        K = length(summands)
        return new{K,N,M}(idx_dict, tuple(summands...))
    end
end
