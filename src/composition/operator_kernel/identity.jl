# Identity-specific overrides for arguments where the abstract-operator
# dispatch in `base.jl` provides a more-specific method. Without these,
# `Identity()(::EvaluationPVCrosscov)` etc. raise a method ambiguity:
# `Identity` is more specific in the operator slot, but the abstract method
# is more specific in the argument slot. Each override below matches the
# corresponding `base.jl` signature exactly and returns the input unchanged.

import AbstractGPs: ZeroMean
import KernelFunctions: KernelSum, ScaledKernel

# Mean
(::Identity)(zm::ZeroMean{T}, args...) where {T} = zm

# Kernels
(::Identity)(k::KernelSum, args...; kwargs...) = k
(::Identity)(k::ScaledKernel, args...; kwargs...) = k
(::Identity)(k::LinearlyScaledKernel, args...; kwargs...) = k

# PV crosscovs (signatures here intentionally take no varargs/kwargs to
# match the abstract dispatches in `base.jl`).
(::Identity)(pv::EvaluationPVCrosscov) = pv
(::Identity)(pv::StackedPVCrosscov) = pv
(::Identity)(pv::AbstractSumPVCrosscov) = pv
(::Identity)(pv::ConstantScaledPVCrosscov) = pv
