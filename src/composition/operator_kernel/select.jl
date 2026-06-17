# How `Select` applies to multi-output kernels → pins one output argument.
#
# The crosscov case (`Select(p)(::EvaluationPVCrosscov)`) is handled by the
# generic operator-on-crosscov method in `base.jl`, which delegates to
# `Select(p)(pv.k; arg = randproc_arg(pv))` — i.e. these same methods.

# First selection: pin one argument of a fresh multi-output kernel. The pinned
# argument carries the bare `Select`; differential/scaling operators accumulate
# onto it later (see `_accumulate_op`), always keeping the `Select` innermost.
function (op::Select)(k::MultiOutputKernel; arg::Integer = 2)
    return TransformedMultiOutputKernel{Int(arg)}(k, op)
end

# The pin and the spatial part are read back off the stored operator: the `Select`
# is always applied first (innermost), so it is the operator itself for a bare pin
# or the first factor of a composition; the spatial part is whatever remains.
pinned_select(op::Select) = op
pinned_select(op::AbstractConcatenatedLinearFunctionOperator) = first(linfuncops(op))
pinned_output(tmk::TransformedMultiOutputKernel) = pinned_select(tmk.op).output

spatial_op(::Select) = Identity()
function spatial_op(op::AbstractConcatenatedLinearFunctionOperator)
    rest = linfuncops(op)[2:end]
    return length(rest) == 1 ? only(rest) : ConcatenatedLinearFunctionOperator(rest)
end
spatial_op(tmk::TransformedMultiOutputKernel) = spatial_op(tmk.op)

# Second selection: pinning the *other* argument determines both outputs, so we
# resolve to the single-output block with the pinned argument's spatial operator
# applied. Re-selecting the already-pinned argument is a usage error — each
# argument is selected exactly once.
function (op::Select)(tmk::TransformedMultiOutputKernel; arg::Integer = 2)
    a = pinned_arg(tmk)
    arg == a && error(
        "argument $arg of this TransformedMultiOutputKernel is already pinned " *
            "(to output $(pinned_output(tmk))); select the other argument to resolve a block",
    )
    p1 = a == 1 ? pinned_output(tmk) : op.output
    p2 = a == 2 ? pinned_output(tmk) : op.output
    return spatial_op(tmk)(_block(tmk.parent, p1, p2); arg = a)
end

# Accumulate an operator onto the pinned argument, keeping the `Select` innermost.
# Operators that carry their own `(op)(::Kernel)` method (e.g. `PartialDerivative`,
# `ConstantScaledLinearFunctionOperator`) forward here from their own files.
_accumulate_op(new_op, tmk::TransformedMultiOutputKernel{K, Arg}) where {K, Arg} =
    TransformedMultiOutputKernel{Arg}(tmk.parent, new_op ∘ tmk.op)

# Direct (KernelFunctions-style) evaluation of a bare pin: supply the free
# argument's output as a `(point, output)` tuple, the pinned side a plain point.
(tmk::TransformedMultiOutputKernel{<:Any, 2, <:Select})((x, p)::Tuple, y) =
    _block(tmk.parent, p, tmk.op.output)(x, y)
(tmk::TransformedMultiOutputKernel{<:Any, 1, <:Select})(x, (y, q)::Tuple) =
    _block(tmk.parent, tmk.op.output, q)(x, y)

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
