using KernelFunctions: Kernel, ZeroKernel

export MultiOutputKernel, BlockDiagonalKernel, n_outputs

"""
    MultiOutputKernel <: Kernel

Abstract supertype for multi-output kernels. A multi-output kernel defines the
covariance between outputs `p` and `q` of a vector-valued GP. Concrete subtypes
answer one query: [`_block`](@ref)`(k, p, q)` — the (single-output) kernel
giving the covariance between output `p` and output `q`.

Outputs are selected with the [`Select`](@ref) operator rather than by encoding
the output index in the input points, so the spatial functionals
(`EvaluationFunctional`, `PartialDerivative`, integrals) never need to know that
multi-output is involved: `EvaluationFunctional(X) ∘ Select(p)` evaluates output
`p` at `X`.
"""
abstract type MultiOutputKernel <: Kernel end

"""
    n_outputs(k::MultiOutputKernel) -> Int

Number of outputs of the multi-output kernel.
"""
function n_outputs end

"""
    _block(k::MultiOutputKernel, p::Int, q::Int) -> Kernel

The single-output kernel describing the covariance between output `p` (first
argument) and output `q` (second argument). For independent outputs this is the
zero kernel whenever `p != q`.
"""
function _block end

"""
    BlockDiagonalKernel(k1, k2, …)

Independent multi-output kernel: output `p` has its own kernel `kₚ` and distinct
outputs are uncorrelated. The joint covariance over outputs is therefore
block-diagonal, with block `p` equal to `kₚ`'s own covariance.

This is the multi-output kernel KernelFunctions.jl is missing — `IndependentMOKernel`
shares a single base kernel across all outputs, whereas this carries a *distinct*
kernel per output.

# Example
```julia
k1 = HalfIntegerMaternKernel(2, 0.3)
k2 = with_lengthscale(SqExponentialKernel(), 0.5)
kmo = BlockDiagonalKernel(k1, k2)
f   = GP(kmo)

ℒ1 = EvaluationFunctional(X1) ∘ Select(1)   # output 1 at X1
ℒ2 = EvaluationFunctional(X2) ∘ Select(2)   # output 2 at X2
fg = FunctionalGaussian(f; a = ℒ1, b = ℒ2)
```
"""
struct BlockDiagonalKernel{KS <: Tuple} <: MultiOutputKernel
    kernels::KS
end

BlockDiagonalKernel(kernels::Kernel...) = BlockDiagonalKernel(kernels)

n_outputs(k::BlockDiagonalKernel) = length(k.kernels)

_block(k::BlockDiagonalKernel, p::Int, q::Int) = p == q ? k.kernels[p] : ZeroKernel()

# KernelFunctions-style evaluation on (x, p) points, for interop / sanity. The
# functional pipeline never hits this — it goes through `Select` + the resolved
# single-output block instead.
function (k::BlockDiagonalKernel)((x, px)::Tuple, (y, py)::Tuple)
    return px == py ? k.kernels[px](x, y) : false
end

"""
    TransformedMultiOutputKernel{Arg}(parent, op)

A [`MultiOutputKernel`](@ref) with an arbitrary linear operator applied to one of
its arguments. Argument `Arg` (`1` or `2`) carries the operator `op`; the other
argument is still a free multi-output process. The output pin is *not* a separate
field — it is carried by `op`: a bare pin is just `op === Select(p)`, so the type
of a pinned-but-untransformed kernel is `TransformedMultiOutputKernel{K, Arg,
<:Select}` and the pinned output is `op.output`. Differential/scaling operators
compose onto `op` (the `Select` stays innermost), so `op` is `spatial ∘ Select(p)`.

This is the transient "half-pinned" representation the two-stage functional
pipeline needs. Operators applied to the pinned argument *accumulate* into `op`
symbolically rather than being applied to the multi-output parent eagerly — so a
`PartialDerivative` is only ever evaluated once the output is pinned and the block
is a single-output kernel, never differentiating every output of the parent at
once. The functional that finally consumes the argument defers the whole thing
into a [`MultiOutputPVCrosscov`](@ref), which resolves to the operator applied to
the single-output block `_block(parent, p₁, p₂)` once the other output is pinned.
"""
struct TransformedMultiOutputKernel{K, Arg, Op} <: Kernel
    parent::K
    op::Op

    # `Arg` is an `Int` value parameter, so it cannot carry a `<: Union{…}` bound
    # the way a type parameter would; the guard is enforced here instead. Because
    # `Arg` is a compile-time constant the branch is constant-folded away.
    function TransformedMultiOutputKernel{K, Arg, Op}(parent::K, op::Op) where {K, Arg, Op}
        (Arg === 1 || Arg === 2) ||
            error("TransformedMultiOutputKernel pinned argument must be 1 or 2, got $Arg")
        return new{K, Arg, Op}(parent, op)
    end
end

TransformedMultiOutputKernel{Arg}(parent::K, op::Op) where {K, Arg, Op} =
    TransformedMultiOutputKernel{K, Arg, Op}(parent, op)

"""
    pinned_arg(tmk::TransformedMultiOutputKernel) -> Int

The kernel argument (`1` or `2`) that `tmk` carries an operator on.
"""
pinned_arg(::TransformedMultiOutputKernel{K, Arg}) where {K, Arg} = Arg
