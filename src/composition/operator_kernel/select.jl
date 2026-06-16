# How `Select` applies to multi-output kernels → pins one output argument.
#
# The crosscov case (`Select(p)(::EvaluationPVCrosscov)`) is handled by the
# generic operator-on-crosscov method in `base.jl`, which delegates to
# `Select(p)(pv.k; arg = randproc_arg(pv))` — i.e. these same methods.

# First selection: pin one argument of a fresh multi-output kernel. The pinned
# argument starts with the identity operator; differential/scaling operators
# accumulate onto it later (see `_accumulate_op`).
function (op::Select)(k::MultiOutputKernel; arg::Integer = 2)
    return TransformedMultiOutputKernel{Int(arg)}(k, op.output, Identity())
end

# Second selection: pinning the *other* argument determines both outputs, so we
# resolve to the single-output block with the pinned argument's stored operator
# applied. Re-selecting the already-pinned argument is a usage error — each
# argument is selected exactly once.
function (op::Select)(tmk::TransformedMultiOutputKernel; arg::Integer = 2)
    a = pinned_arg(tmk)
    arg == a && error(
        "argument $arg of this TransformedMultiOutputKernel is already pinned " *
            "(to output $(tmk.p)); select the other argument to resolve a block",
    )
    p1 = a == 1 ? tmk.p : op.output
    p2 = a == 2 ? tmk.p : op.output
    return tmk.op(_block(tmk.parent, p1, p2); arg = a)
end

# Compose an operator onto the pinned argument of a half-pinned kernel (the
# identity drops out, keeping the common pure-`Select` case operator-free), and
# the corresponding accumulation that returns the updated kernel. Operators that
# carry their own `(op)(::Kernel)` method (e.g. `PartialDerivative`,
# `ConstantScaledLinearFunctionOperator`) forward here from their own files.
_compose_op(new_op, ::Identity) = new_op
_compose_op(new_op, op) = new_op ∘ op

_accumulate_op(new_op, tmk::TransformedMultiOutputKernel{K, Arg}) where {K, Arg} =
    TransformedMultiOutputKernel{Arg}(tmk.parent, tmk.p, _compose_op(new_op, tmk.op))

# Applying `Select` to a single-output kernel is unsupported; it deliberately
# has no method (→ MethodError) rather than a `::Kernel` fallback, which would be
# ambiguous with the generic operator methods for ScaledKernel / KernelSum /
# LinearlyScaledKernel in base.jl.

# Selecting the still-free output of a MultiOutputPVCrosscov pins the remaining
# (process-side) output, so both outputs are now determined: resolve to the
# single-output kernel block and apply the stored functional. For independent
# outputs an off-diagonal block is the zero kernel, which assembles to zeros.
function (op::Select)(pv::MultiOutputPVCrosscov)
    return pv.linfunc(_resolved_block(pv, op.output); arg = randvar_arg(pv))
end
