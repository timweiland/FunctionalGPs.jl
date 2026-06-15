export MultiOutputPVCrosscov

"""
    MultiOutputPVCrosscov{Arg, TK, TL} <: ProcessVectorCrossCovariance

PV crosscov produced when a linear functional is applied to one argument of a
[`MultiOutputKernel`](@ref) whose output on that argument has been pinned with
[`Select`](@ref) — i.e. the crosscov-level analogue of a half-pinned
[`SelectedKernel`](@ref).

A composition such as `EvaluationFunctional(X) ∘ Select(p)` pins output `p` on
one argument (via `Select`) and consumes the spatial part of that same argument
(via the functional), while the *other* argument is left as a full multi-output
process. The selection is embedded directly: the crosscov stores the underlying
multi-output kernel together with the pinned output index `p`, with the argument
it was pinned on carried as the type parameter `Arg`, rather than wrapping a
`SelectedKernel`.

The random-variable side is therefore an ordinary single-output crosscov, while
the process side still carries a free output index. It is resolved by selecting
that output, collapsing to a single-output block via [`Select`](@ref).

# Type parameters
- `Arg`: The kernel argument (1 or 2) carrying the pin and the functional

# Fields
- `k::TK`: The underlying [`MultiOutputKernel`](@ref)
- `p::Int`: The output index pinned on argument `Arg`
- `linfunc::TL`: The linear functional applied to argument `Arg`

# See also
- [`Select`](@ref): Pins the remaining output, resolving the crosscov to a block
- [`MultiOutputKernel`](@ref), [`SelectedKernel`](@ref)
"""
struct MultiOutputPVCrosscov{A, TK <: MultiOutputKernel, TL <: AbstractLinearFunctional} <:
       ProcessVectorCrossCovariance
    k::TK
    p::Int
    linfunc::TL
end

function MultiOutputPVCrosscov{A}(
    k::MultiOutputKernel,
    p::Integer,
    linfunc::AbstractLinearFunctional,
) where {A}
    @assert A ∈ (1, 2) "arg must be 1 or 2, got $arg"
    @assert 1 ≤ p ≤ n_outputs(k) "output $p out of range for $(n_outputs(k))-output kernel"
    return MultiOutputPVCrosscov{A, typeof(k), typeof(linfunc)}(k, Int(p), linfunc)
end

randvar_arg(::MultiOutputPVCrosscov{A}) where {A} = A

# Only the pinned output's functional is fixed, so the random vector is exactly
# the functional's own output. The batch size is flattened to a 1-tuple to match
# the crosscovs this resolves to (e.g. TensorProductCrosscov for a tensor-product
# functional over a separable block), so sums/stacks of resolved and unresolved
# crosscovs agree.
randvar_batch_size(pv::MultiOutputPVCrosscov) = (prod(output_shape(pv.linfunc)),)

# The single-output kernel block coupling the pinned output to the free-side
# output `q`, with `q` placed on the (currently free) process argument.
_resolved_block(pv::MultiOutputPVCrosscov{1}, q::Integer) = _block(pv.k, pv.p, q)
_resolved_block(pv::MultiOutputPVCrosscov{2}, q::Integer) = _block(pv.k, q, pv.p)
