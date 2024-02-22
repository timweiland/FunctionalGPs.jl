export LinearDifferentialOperator

struct LinearDifferentialOperator{K,N,M} <: AbstractSumLinearFunctionOperator{K}
    idx_dict::Dict{K,AbstractVector{NTuple{M,Integer}}}
    summands::Vector{PartialDerivative}

    function LinearDifferentialOperator{N,M}(
        idx_dict::Dict{K,Vector{T}},
    ) where {N,M,K<:Integer,T<:NTuple{M,Integer}}
        if !all(1 <= key <= N for key in keys(idx_dict))
            throw(DomainError("Keys of idx_dict must be between 1 and $N"))
        end
        summands = []
        for (output_idx, multi_idx_list) in idx_dict
            for multi_idx in multi_idx_list
                push!(summands, PartialDerivative{N,M}(output_idx, multi_idx))
            end
        end
        return new{K,N,M}(idx_dict, summands)
    end
end

function LinearDifferentialOperator{N}(idx_dict::Dict{K,Vector{T}}) where {N,M,K<:Integer,T<:NTuple{M,Integer}}
    return LinearDifferentialOperator{N,M}(idx_dict)
end

function LinearDifferentialOperator(idx_dict::Dict{K,Vector{T}}) where {M,K<:Integer,T<:NTuple{M,Integer}}
    return LinearDifferentialOperator{1,M}(idx_dict)
end
