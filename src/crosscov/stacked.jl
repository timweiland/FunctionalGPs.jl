export StackedPVCrosscov
using BlockArrays

"""
    StackedPVCrosscov{T} <: ProcessVectorCrossCovariance

A cross-covariance formed by vertically stacking multiple PV crosscovs.

Use this to combine multiple functionals applied to the same kernel argument,
producing a block-structured random vector.

# Fields
- `pv_crosscovs::Vector{T}`: The crosscovs to stack

# Construction

Typically created via the union operator `∪` on vectors of crosscovs:

```julia
pv_stacked = [pv1, pv2] ∪ [pv3]
```

Or by applying a [`StackedLinearFunctional`](@ref) to a kernel.

# Examples
```julia
julia> k = SqExponentialKernel();
julia> pv1 = EvaluationFunctional([0.0, 0.5])(k);
julia> pv2 = EvaluationFunctional([1.0, 1.5, 2.0])(k);
julia> stacked = StackedPVCrosscov([pv1, pv2]);
julia> randvar_length(stacked)
5
```

# See also
- [`StackedLinearFunctional`](@ref): Functional that creates stacked crosscovs
"""
struct StackedPVCrosscov{T <: ProcessVectorCrossCovariance} <: ProcessVectorCrossCovariance
    pv_crosscovs::Vector{T}

    function StackedPVCrosscov(
            pv_crosscovs::Vector{T},
        ) where {T <: ProcessVectorCrossCovariance}
        @assert all(
            randvar_arg(pv_crosscovs[1]) == randvar_arg(pv_crosscov) for
                pv_crosscov in pv_crosscovs
        )
        return new{T}(pv_crosscovs)
    end
end

function randvar_batch_size(pv::StackedPVCrosscov)
    return (mapreduce(randvar_length, +, pv.pv_crosscovs),)
end
randvar_arg(pv::StackedPVCrosscov) = randvar_arg(pv.pv_crosscovs[1])

function Base.iterate(stacked_pv::StackedPVCrosscov, state = 1)
    return state > length(stacked_pv.pv_crosscovs) ? nothing :
        (stacked_pv.pv_crosscovs[state], state + 1)
end

Base.length(pv::StackedPVCrosscov) = length(pv.pv_crosscovs)

function Base.:(∪)(
        pvs1::Vector{T1},
        pvs2::Vector{T2},
    ) where {T1 <: ProcessVectorCrossCovariance, T2 <: ProcessVectorCrossCovariance}
    return StackedPVCrosscov([pvs1; pvs2])
end
function Base.:(∪)(
        s::StackedPVCrosscov,
        pv::Vector{T},
    ) where {T <: ProcessVectorCrossCovariance}
    return s.pv_crosscovs ∪ pv
end
function Base.:(∪)(
        pv::Vector{T},
        s::StackedPVCrosscov,
    ) where {T <: ProcessVectorCrossCovariance}
    return pv ∪ s.pv_crosscovs
end

function kernelmatrix(pv::StackedPVCrosscov, X::AbstractVector)
    blocks = [kernelmatrix(pv_crosscov, X) for pv_crosscov in pv.pv_crosscovs]
    if randvar_arg(pv) == 1
        # Randvar on rows: blocks have shape (n_randvar_i × length(X))
        # Stack vertically (n×1 block column)
        return mortar(reshape(blocks, :, 1))
    else
        # Randvar on columns: blocks have shape (length(X) × n_randvar_i)
        # Stack horizontally (1×n block row)
        return mortar(reshape(blocks, 1, :))
    end
end

function Base.isequal(pv1::StackedPVCrosscov, pv2::StackedPVCrosscov)
    if length(pv1.pv_crosscovs) != length(pv2.pv_crosscovs)
        return false
    end
    return all(pv1.pv_crosscovs .== pv2.pv_crosscovs)
end

function Base.isapprox(pv1::StackedPVCrosscov, pv2::StackedPVCrosscov)
    if length(pv1.pv_crosscovs) != length(pv2.pv_crosscovs)
        return false
    end
    return all(pv1.pv_crosscovs .≈ pv2.pv_crosscovs)
end
