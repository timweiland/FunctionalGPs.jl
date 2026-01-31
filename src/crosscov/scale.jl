import KernelFunctions: kernelmatrix

export AbstractScaledPVCrosscov, ConstantScaledPVCrosscov

abstract type AbstractScaledPVCrosscov <: ProcessVectorCrossCovariance end
pv_crosscov(op::AbstractScaledPVCrosscov) = op.pv_crosscov
scale(op::AbstractScaledPVCrosscov) = error("scale not implemented for $(typeof(op))")
randvar_arg(op::AbstractScaledPVCrosscov) = randvar_arg(pv_crosscov(op))
randvar_batch_size(op::AbstractScaledPVCrosscov) = randvar_batch_size(pv_crosscov(op))

function (op::AbstractScaledPVCrosscov)(args...; kwargs...)
    return scale(op) * pv_crosscov(op)(args...; kwargs...)
end

function Base.show(io::IO, op::AbstractScaledPVCrosscov)
    return print(io, "$(string(scale(op))) * ($(string(pv_crosscov(op))))")
end

function kernelmatrix(op::AbstractScaledPVCrosscov, x::AbstractVector)
    return scale(op) * kernelmatrix(op.pv_crosscov, x)
end

struct ConstantScaledPVCrosscov <: AbstractScaledPVCrosscov
    pv_crosscov::ProcessVectorCrossCovariance
    scalar::Number
end

scale(op::ConstantScaledPVCrosscov) = op.scalar

Base.:(*)(x::AbstractVector, pv_crosscov::ProcessVectorCrossCovariance) = (length(x) == 1) ? x[1] * pv_crosscov : error("Cannot scale by a vector")
function Base.:(*)(x::Number, pv_crosscov::ProcessVectorCrossCovariance)
    if x == 1
        return pv_crosscov
    end
    return ConstantScaledPVCrosscov(pv_crosscov, x)
end
function Base.:(*)(x::Number, op::ConstantScaledPVCrosscov)
    if x == 1
        return op
    end
    return ConstantScaledPVCrosscov(op.pv_crosscov, x * op.scalar)
end
function Base.:(-)(pv::ProcessVectorCrossCovariance)
    return -1 * pv
end
function Base.:(-)(op::ConstantScaledPVCrosscov)
    return ConstantScaledPVCrosscov(op.pv_crosscov, -op.scalar)
end

function Base.isequal(op1::ConstantScaledPVCrosscov, op2::ConstantScaledPVCrosscov)
    return op1.pv_crosscov == op2.pv_crosscov && op1.scalar == op2.scalar
end

function Base.isapprox(op1::ConstantScaledPVCrosscov, op2::ConstantScaledPVCrosscov)
    return isapprox(op1.pv_crosscov, op2.pv_crosscov) && isapprox(op1.scalar, op2.scalar)
end
