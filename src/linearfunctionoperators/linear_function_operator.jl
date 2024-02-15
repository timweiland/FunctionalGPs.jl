import Base: show
import KernelFunctions: Kernel, KernelTensorProduct

# Define an abstract type for linear operators
abstract type AbstractLinearFunctionOperator end

abstract type AbstractDifferentialOperator <: AbstractLinearFunctionOperator end

abstract type SumLinearFunctionOperator <: AbstractLinearFunctionOperator end
summands(op::SumLinearFunctionOperator) = error("Summands not defined for $(typeof(op))")
function (op::SumLinearFunctionOperator)(args...)
    sum([summand(args...) for summand ∈ summands(op)])
end
function Base.show(io::IO, op::SumLinearFunctionOperator)
    print(io, join(["($(string(summand)))" for summand ∈ summands(op)], " + "))
end


struct PartialDerivative{N, M} <: AbstractDifferentialOperator
    output_idx::Integer
    multi_idx::NTuple{M, Integer}
    order::Integer

    function PartialDerivative{N, M}(output_idx::Integer, multi_idx::NTuple{M, Integer}) where {N, M}
        @assert 1 <= output_idx <= N
        order = sum(multi_idx)
        new(output_idx, multi_idx, order)
    end
end

struct LinearDifferentialOperator{N, M} <: SumLinearFunctionOperator
    idx_dict::Dict{Integer, AbstractVector{NTuple{M, Integer}}}
    summands::Vector{PartialDerivative}

    function LinearDifferentialOperator{N, M}(idx_dict::Dict{K, Vector{T}}) where {N, M, K <: Integer, T <: NTuple{M, Integer}}
        @assert all(1 <= key <= N for key in keys(idx_dict))
        summands = []
        for (output_idx, multi_idx_list) in idx_dict
            for multi_idx in multi_idx_list
                push!(summands, PartialDerivative{N, M}(output_idx, multi_idx))
            end
        end
        new{N, M}(idx_dict, summands)
    end
end
summands(op::LinearDifferentialOperator) = op.summands

UNDERSCORE_CODE_START = 0x2080
POWER_CODE_FOUR = 0x2074

function underscore_char(i::Integer)
    if i >= 10 || i < 0
        return "_$i"
    end
    return Char(UNDERSCORE_CODE_START + i)
end
function power_char(i::Integer)
    if i >= 10 || i < 0
        return "^{$i}"
    end
    if i == 0
        return '⁰'
    elseif i >= 4
        return Char(POWER_CODE_FOUR + i - 4)
    else
        return [Char(0x00B9), Char(0x00B2), Char(0x00B3)][i]
    end
end

function show(io::IO, op::PartialDerivative)
    denominator_str = ""
    for (i, idx) in enumerate(op.multi_idx)
        denominator_str *= "∂x$(underscore_char(i))$(power_char(idx))"
    end
    print(io, "∂$(power_char(op.order))f$(underscore_char(op.output_idx)) / $(denominator_str)")
end

function (op::PartialDerivative{1, M})(k::KernelTensorProduct; kwargs...) where M
    @assert M == length(op.multi_idx)
    ks = Vector{Kernel}(undef, length(op.multi_idx))
    for (i, order) in enumerate(op.multi_idx)
        pd = PartialDerivative{1, 1}(1, (order,))
        ks[i] = pd(k.kernels[i]; kwargs...)
    end
    return KernelTensorProduct(ks)
end

function (op::PartialDerivative{1, 1})(k::CompactPolynomialKernel; first_arg::Bool=true)
    if first_arg
        return derivative(k, op.order, 0)
    else
        return derivative(k, 0, op.order)
    end
end