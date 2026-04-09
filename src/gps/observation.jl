import Distributions: MvNormal
using Distributions: Normal, AbstractMvNormal
using Distributions
using SparseArrays: spzeros, AbstractSparseMatrix
using PDMats: PDSparseMat
using LinearAlgebra
import Base: show

export LinearObservation

# Allow constructing MvNormal with sparse covariance while preserving sparsity
function MvNormal(μ::AbstractVector{<:Real}, Σ::AbstractSparseMatrix{<:Real})
    return MvNormal(μ, PDSparseMat(Σ))
end

"""
    LinearObservation{Tℒ, Tε, Ty}

Represents a noisy observation of a GP through a linear functional.

Given a GP `f`, a linear functional `ℒ`, and observed values `y`, this encodes
the observation model `y = ℒ(f) + ε` where `ε` is additive Gaussian noise.

# Fields
- `linfunc`: The linear functional applied to the GP
- `noise`: The noise distribution (MvNormal or nothing for noise-free)
- `y`: The observed values

See also: [`condition_on_observation`](@ref), [`LinearConditionalGP`](@ref)
"""
struct LinearObservation{Tℒ, Tε, Ty}
    linfunc::Tℒ
    noise::Tε
    y::Ty
end

# Pretty printing
_noise_summary(::Nothing, N) = "none"
_noise_summary(ε::Diagonal, N) = "Diagonal(σ²=$(ε[1, 1]))"
function _noise_summary(ε::AbstractMvNormal, N)
    sparse = ε.Σ isa PDSparseMat
    return sparse ? "MvNormal(n=$(N), sparse)" : "MvNormal(n=$(N))"
end

function show(io::IO, obs::LinearObservation)
    N = Base.length(obs.y)
    print(io, "LinearObservation(")
    print(io, "linfunc = ", string(obs.linfunc), ", ")
    print(io, "y = ", typeof(obs.y), size(obs.y), ", ")
    print(io, "noise = ", _noise_summary(obs.noise, N))
    return print(io, ")")
end

Base.length(obs::LinearObservation) = Base.length(obs.y)

"""
    LinearObservation(linfunc, y; noise=1e-8)

Create a linear observation of `linfunc(f)` with observed values `y` and additive Gaussian noise.

# Arguments
- `linfunc::AbstractLinearFunctional`: The linear functional (e.g., evaluation, derivative, integral)
- `y::AbstractArray`: The observed values

# Keyword Arguments
- `noise`: Observation noise specification (default: `1e-8` for numerical stability)
  - `nothing`: noise-free observation
  - `Real`: scalar variance σ², applied as sparse diagonal σ²I
  - `AbstractVector`: elementwise variances as sparse diagonal
  - `AbstractMatrix`: full covariance matrix
  - `AbstractMvNormal`: full noise distribution (should have zero mean)

# Example
```julia
using FunctionalGPs, AbstractGPs

# Observe function values at points
X = [0.0, 0.5, 1.0]
y = sin.(X)
obs = LinearObservation(EvaluationFunctional(X), y; noise=0.01)

# Observe derivative at a point
∂x = PartialDerivative((1,))
ℒ = EvaluationFunctional([0.5]) ∘ ∂x
obs_deriv = LinearObservation(ℒ, [1.0]; noise=1e-6)
```
"""
function LinearObservation(
        linfunc::AbstractLinearFunctional,
        y::AbstractArray;
        noise::Union{
            Nothing,
            Real,
            AbstractVector{<:Real},
            AbstractMatrix{<:Real},
            AbstractMvNormal,
        } = 1.0e-8,
    )
    N = Base.length(y)
    ε = _noise_to_mvn(noise, N)
    return LinearObservation{typeof(linfunc), typeof(ε), typeof(y)}(linfunc, ε, y)
end

function noise_mean(obs)
    return Distributions.mean(obs.noise)
end

function noise_mean(obs::LinearObservation{TL, Nothing}) where {TL}
    return zeros(Base.length(obs.y))
end

function noise_cov(obs)
    return cov(obs.noise)
end

function noise_cov(obs::LinearObservation{TL, MvNormal}) where {TL}
    return obs.noise.Σ.mat
end

function noise_cov(::LinearObservation{TL, Nothing}) where {TL}
    return nothing
end

function noise_cov(obs::LinearObservation{TL, <:Diagonal}) where {TL}
    return obs.noise
end

function noise_mean(obs::LinearObservation{TL, <:Diagonal}) where {TL}
    return zeros(Base.length(obs.y))
end

function _noise_to_mvn(::Nothing, ::Integer)
    return nothing
end
function _noise_to_mvn(noise::Real, N::Integer)
    return Diagonal(fill(noise, N))
end
function _noise_to_mvn(noise::AbstractVector{<:Real}, N::Integer)
    if length(noise) != N
        throw(DimensionMismatch("noise vector length $(length(noise)) ≠ length(y) $N"))
    end
    Σ = Diagonal(noise * ones(N))
    return MvNormal(zeros(N), Σ)
end
function _noise_to_mvn(Σ::AbstractMatrix{<:Real}, N::Integer)
    if size(Σ, 1) != N || size(Σ, 2) != N
        throw(DimensionMismatch("noise covariance must be $N×$N, got $(size(Σ))"))
    end
    return MvNormal(zeros(N), Σ)
end
function _noise_to_mvn(ε::AbstractMvNormal, N::Integer)
    if length(ε) != N
        throw(DimensionMismatch("noise distribution length $(length(ε)) ≠ length(y) $N"))
    end
    return ε
end
