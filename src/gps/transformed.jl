import AbstractGPs: FiniteGP, AbstractGP, MeanFunction
import Random: AbstractRNG
using Memoize
using LinearAlgebra
import Distributions: MvNormal

_maybe_noise_cov(noise_cov, n) = noise_cov === nothing ? nothing : noise_cov

function _add_noise_cov(Σ::AbstractMatrix, noise_cov)
    if noise_cov === nothing
        return Σ
    elseif noise_cov isa Real
        return Σ + (noise_cov * I(size(Σ, 1)))
    else
        return Σ + noise_cov
    end
end

"""
    (ℒ::AbstractLinearFunctional)(f::AbstractGP; noise=nothing) -> MvNormal

Apply a linear functional to a GP, returning the resulting multivariate normal distribution.

Since `ℒ(f)` is a linear transformation of a Gaussian process, the result is Gaussian with:
- Mean: `ℒ(m)` where `m` is the GP mean function
- Covariance: `ℒ(ℒ(k))` where `k` is the GP kernel

# Arguments
- `ℒ`: A linear functional (evaluation, integral, derivative, or composition)
- `f`: A Gaussian process (`GP` or `LinearConditionalGP`)

# Keyword Arguments
- `noise`: Optional observation noise (scalar variance or covariance matrix)

# Example
```julia
using FunctionalGPs, AbstractGPs

k = Matern52Kernel()
f = GP(k)

# Distribution of function values at points
X = [0.0, 0.5, 1.0]
dist = EvaluationFunctional(X)(f)  # MvNormal

# Distribution of integral over [0, 1]
ℒ = VectorizedLebesgueIntegral(ClosedInterval(0.0, 1.0))
integral_dist = ℒ(f; noise=1e-6)
```
"""
function (ℒ::AbstractLinearFunctional)(
        f::AbstractGP;
        noise_cov::Union{Nothing, Real, AbstractMatrix} = nothing,
        noise::Union{Nothing, Real, AbstractMatrix} = nothing,
    )
    μ = vec(ℒ(f.mean))
    Σ = ℒ(ℒ(f.kernel))
    Σ = (Σ + Σ') / 2
    Σ = _add_noise_cov(Σ, noise_cov === nothing ? noise : noise_cov)
    return MvNormal(μ, Σ)
end

# Method for posterior GP: computes ℒ applied to the conditioned GP
function (ℒ::AbstractLinearFunctional)(
        f::LinearConditionalGP;
        noise_cov::Union{Nothing, Real, AbstractMatrix} = nothing,
        noise::Union{Nothing, Real, AbstractMatrix} = nothing,
    )
    # Posterior mean: ℒ(m_prior) + cov(ℒ, ℒ_obs) * G^{-1} (y - μ - ℒ_obs(m_prior))
    μ_prior = vec(ℒ(f.prior.mean))
    xkℒs = ℒ(f.kℒs)
    μ = μ_prior + xkℒs * representer_weights(f)

    # Posterior covariance: cov(ℒ, ℒ) - cov(ℒ, ℒ_obs) G^{-1} cov(ℒ, ℒ_obs)'
    Σ_prior = ℒ(ℒ(f.prior.kernel))
    C = G_chol(f)
    Σ_post = Σ_prior - xkℒs * (C \ Array(xkℒs'))
    Σ_post = (Σ_post + Σ_post') / 2
    Σ_post = _add_noise_cov(Σ_post, noise_cov === nothing ? noise : noise_cov)
    return MvNormal(μ, Σ_post)
end
