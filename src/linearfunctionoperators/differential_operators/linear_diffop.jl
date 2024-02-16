export LinearDifferentialOperator

struct LinearDifferentialOperator{N,M} <: SumLinearFunctionOperator
    idx_dict::Dict{Integer,AbstractVector{NTuple{M,Integer}}}
    summands::Vector{PartialDerivative}

    function LinearDifferentialOperator{N,M}(
        idx_dict::Dict{K,Vector{T}},
    ) where {N,M,K<:Integer,T<:NTuple{M,Integer}}
        @assert all(1 <= key <= N for key in keys(idx_dict))
        summands = []
        for (output_idx, multi_idx_list) in idx_dict
            for multi_idx in multi_idx_list
                push!(summands, PartialDerivative{N,M}(output_idx, multi_idx))
            end
        end
        return new{N,M}(idx_dict, summands)
    end
end
