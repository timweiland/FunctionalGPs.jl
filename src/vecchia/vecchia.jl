export vecchia, nameview, OrderingStrategy

"""
    OrderingStrategy

Symbol-valued strategy controlling block ordering inside [`vecchia`](@ref).
The recognised values are:

| Symbol | Meaning |
|--------|---------|
| `:integrals_coarsest` | Place integral blocks rightmost (coarsest), evaluations next, derivatives leftmost (finest). Recommended default for finite-volume / mixed-observation models. |
| `:evaluations_coarsest` | Evaluations rightmost. |
| `:derivatives_coarsest` | Derivatives rightmost. |
| `:natural` | Use the order in which blocks were declared on the `FunctionalGaussian`. |

The numerical priority maps live in [`ORDERING_PRIORITIES`](@ref).
"""
const OrderingStrategy = Symbol

"""
    ORDERING_PRIORITIES

Per-strategy priority maps from [`FunctionalCategory`](@ref) to integer
priority. Smaller numbers are placed leftmost (finest) in the Cholesky
ordering; larger numbers are coarsest. `OTHER` shares its priority with
`DERIVATIVE` so that unknown functionals are conservatively treated as fine.
"""
const ORDERING_PRIORITIES = Dict(
    :integrals_coarsest => Dict(
        DERIVATIVE => 1,
        OTHER => 1,
        EVALUATION => 2,
        FACE_INTEGRAL => 3,
        INTEGRAL => 4,
    ),
    :evaluations_coarsest => Dict(
        DERIVATIVE => 1,
        OTHER => 1,
        FACE_INTEGRAL => 2,
        INTEGRAL => 3,
        EVALUATION => 4,
    ),
    :derivatives_coarsest => Dict(
        INTEGRAL => 1,
        FACE_INTEGRAL => 2,
        EVALUATION => 3,
        DERIVATIVE => 4,
        OTHER => 4,
    ),
)

"""
    vecchia(fg::FunctionalGaussian; ρ = 2.0, λ = 1.5, ordering = :integrals_coarsest)

Build a Gaussian Markov Random Field that approximates `fg` via a
KL-optimal sparse Cholesky (Vecchia approximation).

The block structure of `fg` is preserved end-to-end: the resulting GMRF
exposes the same named blocks as `fg` (`:y`, `:dy`, etc.) so downstream
inference (e.g. INLA) can refer to them by name.

This entry point is implemented in a package extension that loads when
`GaussianMarkovRandomFields` is also `using`'d. The signature, ordering
strategies, and per-functional categorisation live in `FunctionalGPs` proper
so that types and policies are owned by the package whose objects are being
classified.

# Keyword arguments
- `ρ`: sparsity radius for the KL-optimal Cholesky (larger ⇒ denser, more
  accurate).
- `λ`: supernodal clustering threshold; pass `nothing` for the simplicial
  (column-by-column) factorisation.
- `ordering`: a value of [`OrderingStrategy`](@ref).

Without `using GaussianMarkovRandomFields`, calling `vecchia(fg)` errors
with a clear "extension not loaded" message.
"""
function vecchia end

"""
    nameview(g, x) -> NamedTuple

Return a NamedTuple of views into the latent vector `x`, one per named block
of the structured Gaussian `g`. Implementations live in package extensions
(currently `FunctionalGPsGMRFsExt` for `MetaGMRF{NamedBlockMetadata}` produced
by `vecchia(fg)`).
"""
function nameview end
