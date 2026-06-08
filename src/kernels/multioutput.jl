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

# Differentiating acts on the spatial part of every output block; the block
# structure (and hence independence) is preserved. Off-diagonal blocks stay
# implicitly zero, so they are never differentiated.
derivative(k::BlockDiagonalKernel, n::Int, m::Int) =
    BlockDiagonalKernel(map(ki -> derivative(ki, n, m), k.kernels))

# KernelFunctions-style evaluation on (x, p) points, for interop / sanity. The
# functional pipeline never hits this — it goes through `Select` + the resolved
# single-output block instead.
function (k::BlockDiagonalKernel)((x, px)::Tuple, (y, py)::Tuple)
    return px == py ? k.kernels[px](x, y) : false
end

"""
    SelectedKernel(parent, pin1, pin2)

A multi-output kernel with zero, one, or both output arguments pinned to a
concrete output index (`pin1` for argument 1, `pin2` for argument 2; `nothing`
means not yet pinned). Produced by applying [`Select`](@ref) to a
[`MultiOutputKernel`](@ref).

This is the transient "half-pinned" representation the two-stage functional
pipeline needs: the first functional pins one argument (and may differentiate
it), and the second pins the other. Once both are pinned, every downstream
operation forwards to the resolved single-output block (`_block(parent, pin1,
pin2)`), so all the existing single-output fast paths fire unchanged.
"""
struct SelectedKernel{K, P1, P2} <: Kernel
    parent::K
    pin1::P1
    pin2::P2
end

function _resolve(sk::SelectedKernel)
    (sk.pin1 === nothing || sk.pin2 === nothing) && error(
        "SelectedKernel is not fully resolved (pins: $(sk.pin1), $(sk.pin2)); " *
            "both kernel arguments must be selected before a covariance can be assembled",
    )
    return _block(sk.parent, sk.pin1, sk.pin2)
end

# Differentiation pushes into the parent block-diagonal kernel and keeps the
# pins; once both pins resolve, this is `derivative(k_p, n, m)` for the surviving
# block.
derivative(sk::SelectedKernel, n::Int, m::Int) =
    SelectedKernel(derivative(sk.parent, n, m), sk.pin1, sk.pin2)

(sk::SelectedKernel)(x, y) = _resolve(sk)(x, y)

kernel_evaluate_evaluate(sk::SelectedKernel, X) = kernel_evaluate_evaluate(_resolve(sk), X)
kernel_evaluate_evaluate(sk::SelectedKernel, X1, X2) =
    kernel_evaluate_evaluate(_resolve(sk), X1, X2)

# Zero-block short-circuit: a cross-output block (independent outputs) resolves
# to the zero kernel, which assembles directly to a zero matrix of the right
# shape — no kernel arithmetic.
kernel_evaluate_evaluate(::ZeroKernel, X) = zeros(length(X), length(X))
kernel_evaluate_evaluate(::ZeroKernel, X1, X2) = zeros(length(X1), length(X2))
