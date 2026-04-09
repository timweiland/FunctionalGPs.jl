using ChainRulesCore: ChainRulesCore, @thunk, NoTangent

export gp_nll

"""
    gp_nll(K, y)

Compute the Gaussian process negative log-likelihood:

    NLL = 0.5 * (y'K⁻¹y + log|K| + N⋅log(2π))

where `K` is a positive-definite covariance matrix and `y` is the observation
vector. Provides a ChainRules `rrule` so reverse-mode AD backends can compute
`∂NLL/∂K` analytically as `0.5 * (K⁻¹ - αα')` where `α = K⁻¹y`.
"""
function gp_nll(K::AbstractMatrix, y::AbstractVector)
    N = length(y)
    C = cholesky(Symmetric(K))
    α = C \ y
    return (dot(y, α) + logdet(C) + N * log(2π)) / 2
end

function ChainRulesCore.rrule(::typeof(gp_nll), K::AbstractMatrix, y::AbstractVector)
    N = length(y)
    C = cholesky(Symmetric(K))
    α = C \ y
    nll = (dot(y, α) + logdet(C) + N * log(2π)) / 2

    function gp_nll_pullback(Δ)
        # ∂NLL/∂K = 0.5 * (K⁻¹ - αα')
        # Symmetrize because K is symmetric and the gradient should reflect that
        Kinv = C \ Matrix(I, N, N)
        ∂K = @thunk Δ * (Kinv - α * α') / 2
        return NoTangent(), ∂K, NoTangent()
    end

    return nll, gp_nll_pullback
end
