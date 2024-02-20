import AbstractGPs: AbstractGP, GP, var, mean_and_var
using BlockArrays
using Distributions: Normal, AbstractMvNormal
import Distributions: MvNormal
import Statistics: cov
using SparseArrays: spzeros, AbstractSparseMatrix
using PDMats: PDSparseMat
using Memoize
import Base: rand
import Random: AbstractRNG

export LinearObservation, LinearConditionalGP, condition_on_observation, G_chol, μs, εs, representer_weights, y_vec
export mean, cov

# Hotfixes for sparse covariance matrices in MvNormal...
# Maybe make a PR to Distributions.jl. But the repo is pretty unresponsive atm.
MvNormal(μ::AbstractVector{<:Real}, Σ::AbstractSparseMatrix{<:Real}) = MvNormal(μ, PDSparseMat(Σ))
cov_hotfix(x) = cov(x)
cov_hotfix(ε::MvNormal) = ε.Σ.mat  # Default impl converts to Matrix, destroying sparsity

struct LinearObservation
    ℒ::AbstractLinearFunctional
    ε::Union{Normal,AbstractMvNormal, Nothing}
    y::AbstractArray
end

struct LinearConditionalGP <: AbstractGP
    prior::AbstractGP
    observations::Tuple{Vararg{LinearObservation}}
    kℒs::StackedPVCrosscov
    G::BlockMatrix
    ℒs_mean_tuple::Tuple{Vararg{AbstractArray}}
end

ℒs(f::LinearConditionalGP) = map(o -> o.ℒ, f.observations)
εs(f::LinearConditionalGP) = map(o -> o.ε, f.observations)
μs(f::LinearConditionalGP) = map(mean, εs(f))
μs_vec(f::LinearConditionalGP) = mapreduce(o -> mean(o.ε), vcat, f.observations)
ys(f::LinearConditionalGP) = map(o -> o.y, f.observations)
y_vec(f::LinearConditionalGP) = mapreduce(o -> o.y, vcat, f.observations)
ℒs_mean_vec(f::LinearConditionalGP) = reduce(vcat, f.ℒs_mean_tuple)
G_unblocked(f::LinearConditionalGP) = convert(eltype(f.G.blocks), f.G)
@memoize predictive_residual(f::LinearConditionalGP) = y_vec(f) - (ℒs_mean_vec(f) + μs_vec(f))
@memoize G_chol(f::LinearConditionalGP) = cholesky(G_unblocked(f))
@memoize representer_weights(f::LinearConditionalGP) = G_chol(f) \ predictive_residual(f)

function condition_on_observation(f::GP, observation::LinearObservation)
    ℒ = observation.ℒ
    G_block = ℒ(ℒ(f.kernel))
    if !isnothing(observation.ε)
        G_block += cov_hotfix(observation.ε)
    end
    return LinearConditionalGP(
        f,
        (observation,),
        StackedPVCrosscov([ℒ(f.kernel)]),
        mortar((G_block,)),
        (ℒ(f.mean),),
    )
end

function condition_on_observation(f::AbstractGP, ℒ::AbstractLinearFunctional, y::AbstractArray; noise::Union{Real, Nothing} = nothing)
    if isnothing(noise)
        return condition_on_observation(f, LinearObservation(ℒ, nothing, y))
    end
    N = length(y)
    Σ = spzeros(N, N)
    Σ[diagind(Σ)] .= noise
    return condition_on_observation(f, LinearObservation(ℒ, MvNormal(spzeros(N), Σ), y))
end

function condition_on_observation(f::AbstractGP, X::AbstractVector, y::AbstractArray; noise::Union{Real, Nothing} = nothing)
    return condition_on_observation(f, EvaluationFunctional(X), y, noise=noise)
end

function condition_on_observation(f::LinearConditionalGP, observation::LinearObservation)
    ℒ = observation.ℒ
    ℒ_block = ℒ(ℒ(f.prior.kernel))
    if !isnothing(observation.ε)
        ℒ_block += cov_hotfix(observation.ε)
    end
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

mean(f_cond_eval::FiniteGP{<:LinearConditionalGP}) = mean(f_cond_eval.f.prior(f_cond_eval.x)) + kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x) * representer_weights(f_cond_eval.f)
function cov(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    prior_cov = cov_hotfix(f_cond_eval.f.prior(f_cond_eval.x))
    xkℒs = kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x)
    C = G_chol(f_cond_eval.f)
    Z = C.L \ (xkℒs'[C.p, :])
    return prior_cov - Z' * Z
end
function var(f_cond_eval::FiniteGP{<:LinearConditionalGP})
    prior_var = var(f_cond_eval.f.prior(f_cond_eval.x))
    xkℒs = kernelmatrix(f_cond_eval.f.kℒs, f_cond_eval.x)
    C = G_chol(f_cond_eval.f)
    Z = C.L \ (xkℒs'[C.p, :])
    return prior_var - reshape(sum(Z.^2, dims=1), size(prior_var))
end
mean_and_var(f_cond_eval::FiniteGP{<:LinearConditionalGP}) = (mean(f_cond_eval), var(f_cond_eval))
function rand(rng::AbstractRNG, f_cond_eval::FiniteGP{<:LinearConditionalGP}, n::Int)
    μ = mean(f_cond_eval)
    C = cholesky(cov(f_cond_eval))
    samples = μ .+ Matrix(matrix_sqrt(C) * randn(rng, size(C, 2), n))
    return samples
end
