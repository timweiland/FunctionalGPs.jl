export PartialDerivative

struct PartialDerivative{N, M} <: AbstractDifferentialOperator
    output_idx::Integer
    multi_idx::NTuple{M, Integer}
    order::Integer

    function PartialDerivative{N, M}(
            output_idx::Integer,
            multi_idx::NTuple{M, Integer},
        ) where {N, M}
        if !(1 <= output_idx <= N)
            throw(DomainError(output_idx, "output_idx must be between 1 and $N"))
        end
        order = sum(multi_idx)
        return new(output_idx, multi_idx, order)
    end
end

function PartialDerivative(multi_idx::NTuple{M, Integer}) where {M}
    return PartialDerivative{1, M}(1, multi_idx)
end

function PartialDerivative{N}(
        output_idx::Integer,
        multi_idx::NTuple{M, Integer},
    ) where {N, M}
    return PartialDerivative{N, M}(output_idx, multi_idx)
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
