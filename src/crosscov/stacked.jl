export StackedPVCrosscov
using BlockArrays

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
    return hcat([kernelmatrix(pv_crosscov, X) for pv_crosscov in pv.pv_crosscovs]...)
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
