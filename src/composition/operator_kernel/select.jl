# How `Select` applies to multi-output kernels → pins one output argument.
#
# The crosscov case (`Select(p)(::EvaluationPVCrosscov)`) is handled by the
# generic operator-on-crosscov method in `base.jl`, which delegates to
# `Select(p)(pv.k; arg = randproc_arg(pv))` — i.e. these same methods.

# First selection: pin one argument of a fresh multi-output kernel.
function (op::Select)(k::MultiOutputKernel; arg::Integer = 2)
    return SelectedKernel(k, arg == 1 ? op.output : nothing, arg == 2 ? op.output : nothing)
end

# Second selection: pin the other argument of an already-selected kernel.
function (op::Select)(sk::SelectedKernel; arg::Integer = 2)
    pin1 = arg == 1 ? op.output : sk.pin1
    pin2 = arg == 2 ? op.output : sk.pin2
    return SelectedKernel(sk.parent, pin1, pin2)
end

# Applying `Select` to a single-output kernel is unsupported; it deliberately
# has no method (→ MethodError) rather than a `::Kernel` fallback, which would be
# ambiguous with the generic operator methods for ScaledKernel / KernelSum /
# LinearlyScaledKernel in base.jl.

function (op::Select)(cc::EvaluationPVCrosscov{2, <:SelectedKernel{<:MultiOutputKernel, Nothing, <:Integer}, <:EvaluationFunctional})
    return cc.linfunc(_resolve(op(cc.k; arg = 1)), arg = 2)
end

function (op::Select)(cc::IntegralPVCrosscov{2, <:SelectedKernel{<:MultiOutputKernel, Nothing, <:Integer}, <:Vector})
    return VectorizedLebesgueIntegral(cc.domains)(_resolve(op(cc.k; arg = 1)); arg = 2)
end
