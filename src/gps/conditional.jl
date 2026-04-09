import AbstractGPs: AbstractGP, GP, var, mean_and_var, FiniteGP
using AbstractGPs
using BlockArrays
import Distributions: MvNormal
import Statistics: cov, mean
using Memoize
import Base: rand
import Random: AbstractRNG
using LinearAlgebra

export LinearConditionalGP,
    condition_on_observation,
    G_chol,
    μs,
    εs,
    representer_weights,
    y_vec,
    predictive_residual
export mean, cov

"""
    LinearConditionalGP <: AbstractGP

A Gaussian process conditioned on linear observations.

This type represents a GP posterior after conditioning on one or more `LinearObservation`s.
It stores the prior GP and all observation data needed for efficient posterior inference.

Use [`condition_on_observation`](@ref) to construct instances rather than calling the
constructor directly.

# Posterior Computations
For a `LinearConditionalGP` `f`, you can:
- Evaluate at points: `f(X)` returns a `FiniteGP`
- Get posterior mean: `mean(f(X))`
- Get posterior variance: `var(f(X))`
- Get posterior covariance: `cov(f(X))`
- Sample: `rand(rng, f(X), n)`
- Apply further functionals: `ℒ(f)` returns an `MvNormal`

See also: [`LinearObservation`](@ref), [`condition_on_observation`](@ref)
"""
struct LinearConditionalGP <: AbstractGP
    prior::AbstractGP
    observations::Tuple{Vararg{LinearObservation}}
    kℒs::StackedPVCrosscov
    G::BlockMatrix
    ℒs_mean_tuple::Tuple{Vararg{AbstractArray}}
end

ℒs(f::LinearConditionalGP) = map(o -> o.linfunc, f.observations)

"""
    εs(f::LinearConditionalGP)

Return the tuple of noise distributions from all observations.
"""
εs(f::LinearConditionalGP) = map(o -> o.noise, f.observations)

"""
    μs(f::LinearConditionalGP)

Return the tuple of noise means from all observations.
"""
μs(f::LinearConditionalGP) = map(mean, εs(f))
μs_vec(f::LinearConditionalGP) = mapreduce(o -> noise_mean(o), vcat, f.observations)
ys(f::LinearConditionalGP) = map(o -> o.y, f.observations)

"""
    y_vec(f::LinearConditionalGP)

Return all observed values concatenated into a single vector.
"""
y_vec(f::LinearConditionalGP) = mapreduce(o -> reshape(o.y, :), vcat, f.observations)
ℒs_mean_vec(f::LinearConditionalGP) = mapreduce(m -> reshape(m, :), vcat, f.ℒs_mean_tuple)
function G_unblocked(f::LinearConditionalGP)
    if length(f.G.blocks) == 1
        G = f.G.blocks[1, 1]
    else
        G = convert(eltype(f.G.blocks), f.G)
    end
    # Ensure symmetry for Cholesky factorization
    return Symmetric(G)
end

"""
    predictive_residual(f::LinearConditionalGP)

Compute the predictive residual `y - ℒ(m) - μ_ε` where `y` is observed data,
`ℒ(m)` is the prior mean evaluated through the functional, and `μ_ε` is the noise mean.

This is cached (memoized) for efficiency.
"""
@memoize predictive_residual(f::LinearConditionalGP) =
    y_vec(f) - (ℒs_mean_vec(f) + μs_vec(f))

"""
    G_chol(f::LinearConditionalGP)

Return the Cholesky factorization of the Gram matrix `G = ℒ(ℒ(k)) + Σ_ε`.

This is the core matrix needed for posterior computations and is cached (memoized).
"""
@memoize G_chol(f::LinearConditionalGP) = cholesky(G_unblocked(f))

"""
    representer_weights(f::LinearConditionalGP)

Compute the representer weights `G⁻¹(y - ℒ(m) - μ_ε)` used for posterior mean computation.

The posterior mean at new points is `m(x) + k(x, ℒ) * representer_weights(f)`.
This is cached (memoized) for efficiency.
"""
@memoize representer_weights(f::LinearConditionalGP) = G_chol(f) \ predictive_residual(f)

"""
    condition_on_observation(f::AbstractGP, observation::LinearObservation) -> LinearConditionalGP
    condition_on_observation(f::AbstractGP, ℒ, y; noise=nothing) -> LinearConditionalGP
    condition_on_observation(f::AbstractGP, X::AbstractVector, y; noise=nothing) -> LinearConditionalGP

Condition a GP on observations through a linear functional.

Returns a `LinearConditionalGP` representing the posterior distribution.

# Methods
- `condition_on_observation(f, obs)`: Condition on a `LinearObservation`
- `condition_on_observation(f, ℒ, y; noise)`: Condition on `ℒ(f) = y` with optional noise
- `condition_on_observation(f, X, y; noise)`: Shorthand for point evaluation at `X`

Multiple observations can be added sequentially by calling `condition_on_observation`
on an existing `LinearConditionalGP`.

# Example
```julia
using FunctionalGPs, AbstractGPs

k = Matern52Kernel()
f = GP(k)

# Condition on function values
X = [0.0, 0.5, 1.0]
y = sin.(X)
f_post = condition_on_observation(f, X, y; noise=0.01)

# Add derivative observation
∂x = PartialDerivative((1,))
ℒ = EvaluationFunctional([0.25]) ∘ ∂x
f_post2 = condition_on_observation(f_post, ℒ, [0.5]; noise=1e-6)

# Compute posterior at new points
X_new = 0:0.1:1
μ = mean(f_post2(X_new))
σ² = var(f_post2(X_new))
```
"""
function condition_on_observation(f::GP, observation::LinearObservation)
    ℒ = observation.linfunc
    G_block = ℒ(ℒ(f.kernel))
    eps_cov = noise_cov(observation)
    if !isnothing(eps_cov)
        G_block += eps_cov
    end
    # G_block = Symmetric(G_block)

    return LinearConditionalGP(
        f,
        (observation,),
        StackedPVCrosscov([ℒ(f.kernel)]),
        mortar((G_block,)),
        (ℒ(f.mean),),
    )
end

function condition_on_observation(
        f::AbstractGP,
        ℒ::AbstractLinearFunctional,
        y::AbstractArray;
        noise::Union{Real, Nothing} = nothing,
    )
    return condition_on_observation(f, LinearObservation(ℒ, y; noise = noise))
end

function condition_on_observation(
        f::AbstractGP,
        X::AbstractVector,
        y::AbstractArray;
        noise::Union{Real, Nothing} = nothing,
    )
    return condition_on_observation(f, EvaluationFunctional(X), y; noise = noise)
end

function condition_on_observation(f::LinearConditionalGP, observation::LinearObservation)
    ℒ = observation.linfunc
    ℒ_block = ℒ(ℒ(f.prior.kernel))
    eps_cov = noise_cov(observation)
    if !isnothing(eps_cov)
        ℒ_block += eps_cov
    end
    ℒ_block = (ℒ_block + ℒ_block') / 2
    off_diagonal_blocks = ℒ(f.kℒs)

    N_blocks_prior = length(f.observations)
    new_block_type = typeof(ℒ_block)
    existing_block_type = eltype(f.G.blocks)
    block_type = new_block_type <: existing_block_type ? existing_block_type :
        existing_block_type <: new_block_type ? new_block_type :
        AbstractMatrix{eltype(ℒ_block)}
    G_blocks = Array{block_type}(undef, N_blocks_prior + 1, N_blocks_prior + 1)
    G_blocks[1:N_blocks_prior, 1:N_blocks_prior] = f.G.blocks
    G_blocks[end, 1:N_blocks_prior] = off_diagonal_blocks.blocks
    G_blocks[1:N_blocks_prior, end] = map(adjoint, off_diagonal_blocks.blocks)
    G_blocks[end, end] = ℒ_block

    return LinearConditionalGP(
        f.prior,
        (f.observations..., observation),
        f.kℒs ∪ [ℒ(f.prior.kernel)],
        mortar(G_blocks),
        (f.ℒs_mean_tuple..., ℒ(f.prior.mean)),
    )
end

function mean(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    return AbstractGPs.mean(f_cond_eval.f.prior(f_cond_eval.x)) +
        Vector(
        kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x) *
            representer_weights(f_cond_eval.f)
    )
end
function cov(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    prior_cov = kernel_evaluate_evaluate(f_cond_eval.f.prior.kernel, f_cond_eval.x)
    xkℒs = Matrix(kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x))
    C = G_chol(f_cond_eval.f)
    solve = C \ xkℒs'
    cov_update = xkℒs * solve
    return prior_cov - cov_update
end
function var(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    prior_var = diag(kernel_evaluate_evaluate(f_cond_eval.f.prior.kernel, f_cond_eval.x))
    xkℒs = Matrix(kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x))
    C = G_chol(f_cond_eval.f)
    solve = C \ xkℒs'
    diag_update = vec(sum(xkℒs .* transpose(solve); dims = 2))
    return prior_var - diag_update
end
function mean_and_var(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    return (mean(f_cond_eval), var(f_cond_eval))
end
function rand(rng::AbstractRNG, f_cond_eval::FiniteGP{<:LinearConditionalGP}, n::Int)
    μ = mean(f_cond_eval)
    C = cholesky(cov(f_cond_eval))
    samples = μ .+ Matrix(matrix_sqrt(C) * randn(rng, size(C, 2), n))
    return samples
end
