export PartialDerivative
using KernelFunctions

struct PartialDerivative{N,M} <: AbstractDifferentialOperator
    output_idx::Integer
    multi_idx::NTuple{M,Integer}
    order::Integer

    function PartialDerivative{N,M}(
        output_idx::Integer,
        multi_idx::NTuple{M,Integer},
    ) where {N,M}
        @assert 1 <= output_idx <= N
        order = sum(multi_idx)
        return new(output_idx, multi_idx, order)
    end
end

function PartialDerivative(
    multi_idx::NTuple{M,Integer},
) where {M}
    return PartialDerivative{1,M}(1, multi_idx)
end

function PartialDerivative{N}(
    output_idx::Integer,
    multi_idx::NTuple{M,Integer},
) where {N,M}
    return PartialDerivative{N,M}(output_idx, multi_idx)
end

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

function Base.show(io::IO, op::PartialDerivative)
    denominator_str = ""
    for (i, idx) in enumerate(op.multi_idx)
        denominator_str *= "∂x$(underscore_char(i))$(power_char(idx))"
    end
    return print(
        io,
        "∂$(power_char(op.order))f$(underscore_char(op.output_idx)) / $(denominator_str)",
    )
end

function (op::PartialDerivative{1,M})(k::KernelTensorProduct; kwargs...) where {M}
    @assert M == length(op.multi_idx)
    ks = Vector{Kernel}(undef, length(op.multi_idx))
    for (i, order) in enumerate(op.multi_idx)
        pd = PartialDerivative{1,1}(1, (order,))
        ks[i] = pd(k.kernels[i]; kwargs...)
    end
    return KernelTensorProduct(ks)
end

function (op::PartialDerivative{1,1})(
    k::Union{CompactPolynomialKernel,CompactSignedPolynomialKernel};
    arg::Integer = 1,
)
    if arg ∉ [0, 1]
        return error("arg must be 0 or 1")
    end
    if arg == 0
        return derivative(k, op.order, 0)
    else
        return derivative(k, 0, op.order)
    end
end
