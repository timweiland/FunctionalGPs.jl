import AbstractGPs: AbstractGP, GP, var, mean_and_var, FiniteGP
using AbstractGPs
using BlockArrays
import Distributions: MvNormal
import Statistics: cov
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

struct LinearConditionalGP <: AbstractGP
    prior::AbstractGP
    observations::Tuple{Vararg{LinearObservation}}
    kℒs::StackedPVCrosscov
    G::BlockMatrix
    ℒs_mean_tuple::Tuple{Vararg{AbstractArray}}
end

ℒs(f::LinearConditionalGP) = map(o -> o.linfunc, f.observations)
εs(f::LinearConditionalGP) = map(o -> o.noise, f.observations)
μs(f::LinearConditionalGP) = map(mean, εs(f))
μs_vec(f::LinearConditionalGP) = mapreduce(o -> noise_mean(o), vcat, f.observations)
ys(f::LinearConditionalGP) = map(o -> o.y, f.observations)
y_vec(f::LinearConditionalGP) = mapreduce(o -> reshape(o.y, :), vcat, f.observations)
ℒs_mean_vec(f::LinearConditionalGP) = mapreduce(m -> reshape(m, :), vcat, f.ℒs_mean_tuple)
function G_unblocked(f::LinearConditionalGP)
    if length(f.G.blocks) == 1
        return f.G.blocks[1, 1]
    else
        convert(eltype(f.G.blocks), f.G)
    end
end
@memoize predictive_residual(f::LinearConditionalGP) =
    y_vec(f) - (ℒs_mean_vec(f) + μs_vec(f))
@memoize G_chol(f::LinearConditionalGP) = cholesky(G_unblocked(f))
@memoize representer_weights(f::LinearConditionalGP) = G_chol(f) \ predictive_residual(f)

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
    G_blocks = Array{typeof(ℒ_block)}(undef, N_blocks_prior + 1, N_blocks_prior + 1)
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
        kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x) *
        representer_weights(f_cond_eval.f)
end
function cov(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    prior_cov = f_cond_eval.f.prior(f_cond_eval.x)
    xkℒs = kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x)
    C = G_chol(f_cond_eval.f)
    solve = C \ xkℒs'
    cov_update = xkℒs * solve
    return prior_cov - cov_update
end
function var(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    prior_var = var(f_cond_eval.f.prior(f_cond_eval.x))
    xkℒs = kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x)
    C = G_chol(f_cond_eval.f)
    solve = C \ xkℒs'
    diag_update = sum(xkℒs .* transpose(solve); dims = 2)
    return prior_var - reshape(vec(diag_update), size(prior_var))
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
