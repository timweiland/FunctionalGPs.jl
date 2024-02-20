import AbstractGPs: FiniteGP, AbstractGP, MeanFunction
import Statistics: mean, cov
import Random: AbstractRNG
using Memoize

export LinfctlTransformedGP

struct LinfctlTransformedGP{Tf,TΣy}
    f::Tf
    ℒ::AbstractLinearFunctionOperator
    ℒμ::AbstractArray
    ℒkℒ′::AbstractArray
    Σy::TΣy
end

function (ℒ::AbstractLinearFunctional)(f::AbstractGP; noise::TΣy = 0) where {TΣy}
    return LinfctlTransformedGP(f, ℒ, ℒ(f.mean), ℒ(ℒ(f.kernel)), noise)
end

mean(ℒf::LinfctlTransformedGP) = ℒf.ℒμ

@memoize cov(ℒf::LinfctlTransformedGP) = ℒf.ℒkℒ′ .+ ℒf.Σy
@memoize cholesky_factor(ℒf::LinfctlTransformedGP) = cholesky(cov(ℒf))

function Base.rand(rng::AbstractRNG, ℒf::LinfctlTransformedGP, n::Int)
    μ = mean(ℒf)
    C = cholesky_factor(ℒf)
    samples = μ .+ Matrix(matrix_sqrt(C) * randn(rng, size(C, 2), n))
    return samples
end
