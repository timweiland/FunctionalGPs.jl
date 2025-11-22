export RadialCovarianceFunction1D_Identity_LebesgueIntegral,
    radial_covfunc_one_sided_integral, radial_covfunc_two_sided_integral
export integrate_radial

abstract type RadialCovarianceFunction1D_Identity_LebesgueIntegral <:
ProcessVectorCrossCovariance end

covfunc(pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral) = pv.covfunc
domains(
    pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral,
)::AbstractVector{Interval} = pv.domains
function randvar_batch_size(pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral)
    return size(pv.domains)
end
randvar_arg(pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral) = pv.randvar_arg

_r(x::Real, y::Real; ℓ::Real) = abs(x - y) / ℓ

function radial_covfunc_one_sided_integral(
        a::Real,
        b::Real,
        x::Real;
        anti_derivative::Base.Callable,
        ℓ::Real,
    )
    a_sgn = (a < x) ? -1 : 1
    b_sgn = (b < x) ? -1 : 1
    return ℓ * (
        b_sgn * anti_derivative(_r(x, b; ℓ = ℓ)) -
            a_sgn * anti_derivative(_r(x, a; ℓ = ℓ))
    )
end

function radial_covfunc_two_sided_integral(
        a::Real,
        b::Real,
        c::Real,
        d::Real;
        anti_derivative::Base.Callable,
        ℓ::Real,
    )
    K = (x, y) -> anti_derivative(_r(x, y; ℓ = ℓ))
    return ℓ^2 * (K(b, c) - K(a, c) - K(b, d) + K(a, d))
end

"""
    _interval_bounds_matrices(domains, ::Type{T})

Return matrices of lower and upper bounds (each `n×1`) for the provided
interval vector, converted to element type `T`.
"""
function _interval_bounds_matrices(domains, ::Type{T}) where {T <: AbstractFloat}
    n = length(domains)
    lowers = Matrix{T}(undef, n, 1)
    uppers = Matrix{T}(undef, n, 1)
    @inbounds for i in 1:n
        d = domains[i]
        lowers[i, 1] = T(d.lower)
        uppers[i, 1] = T(d.upper)
    end
    return lowers, uppers
end

"""
    _extract_lengthscale(ℓ)

Return the scalar lengthscale from a 1D kernel specification.
"""
_extract_lengthscale(ℓ) = only(ℓ)

"""
    _lazy_radial_integral_evaluation_matrix(pv, X)

Construct a lazy one-sided radial integral matrix when the kernel supports the
stationary signed representation. Returns `nothing` if the fast path is
unavailable.
"""
function _lazy_radial_integral_evaluation_matrix(
        pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral,
        X::AbstractVector,
    )
    domains_vec = domains(pv)
    isempty(domains_vec) && return nothing
    base_type = promote_type(eltype(X), typeof(domains_vec[1].lower), typeof(domains_vec[1].upper))
    T = float(base_type)
    T <: AbstractFloat || return nothing
    k = covfunc(pv)
    ℓ_val = _extract_lengthscale(k.lengthscales)
    ℓ_val isa Number || return nothing
    lowers, uppers = _interval_bounds_matrices(domains_vec, T)
    points = reshape(T.(X), :, 1)
    anti = radial_antiderivative(k, Val(1))
    ℓT = T(ℓ_val)
    signed_map = let anti = anti, ℓT = ℓT, zero_T = zero(T), one_T = one(T)
        (r2, s) -> begin
            τ = sqrt(max(r2, zero_T))
            s_eff = iszero(s) ? one_T : s
            return s_eff * ℓT * anti(τ)
        end
    end
    scales = T[inv(ℓT)]
    upper_matrix = SignedStationaryKernelMatrix(uppers, points, signed_map; scales = scales)
    lower_matrix = SignedStationaryKernelMatrix(lowers, points, signed_map; scales = scales)
    return upper_matrix - lower_matrix
end

"""
    _lazy_radial_integral_integral_matrix(k, domains1, domains2)

Assemble the two-sided radial integral matrix as a lazy combination of
stationary blocks. Returns `nothing` when the fast path is not applicable.
"""
function _lazy_radial_integral_integral_matrix(k, domains1, domains2)
    (isempty(domains1) || isempty(domains2)) && return nothing
    base_type = promote_type(
        typeof(domains1[1].lower),
        typeof(domains1[1].upper),
        typeof(domains2[1].lower),
        typeof(domains2[1].upper),
    )
    T = float(base_type)
    T <: AbstractFloat || return nothing
    ℓ_val = _extract_lengthscale(k.lengthscales)
    ℓ_val isa Number || return nothing
    lowers1, uppers1 = _interval_bounds_matrices(domains1, T)
    lowers2, uppers2 = _interval_bounds_matrices(domains2, T)
    anti = radial_antiderivative(k, Val(2))
    ℓT = T(ℓ_val)
    radial_map = let anti = anti, ℓ2 = ℓT^2, zero_T = zero(T)
        r2 -> begin
            τ = sqrt(max(r2, zero_T))
            return ℓ2 * anti(τ)
        end
    end
    scales = T[inv(ℓT)]
    term_bc = StationaryKernelMatrix(uppers1, lowers2, radial_map; scales = scales)
    term_ac = -1.0 * StationaryKernelMatrix(lowers1, lowers2, radial_map; scales = scales)
    term_bd = -1.0 * StationaryKernelMatrix(uppers1, uppers2, radial_map; scales = scales)
    term_ad = StationaryKernelMatrix(lowers1, uppers2, radial_map; scales = scales)
    return ApplyArray(+, term_bc, term_ac, term_bd, term_ad)
    #return ((term_bc - term_ac) - term_bd) + term_ad
end

# TODO: Test me later, e.g. for the Matern
function kernelmatrix(
        pv::RadialCovarianceFunction1D_Identity_LebesgueIntegral,
        X::AbstractVector,
    )
    anti = radial_antiderivative(covfunc(pv), Val(1))
    ℓ = covfunc(pv).lengthscales

    lower_bounds = map(domain -> domain.lower, domains(pv))
    upper_bounds = map(domain -> domain.upper, domains(pv))

    lower_bounds, X = reshape_product_broadcast(lower_bounds, X)
    upper_bounds = reshape(upper_bounds, size(lower_bounds)...)

    eval_fn =
        (a, b, x) ->
    radial_covfunc_one_sided_integral(a, b, x; anti_derivative = anti, ℓ = only(ℓ))
    res = eval_fn.(lower_bounds, upper_bounds, X)
    res = reshape(res, randvar_length(pv), :)
    if randvar_arg(pv) == 2
        res = res'
    end
    return res
end

function integrate_radial(
        k,
        domains1,
        domains2,
    )
    anti = radial_antiderivative(k, Val(2))
    ℓ = k.lengthscales

    res = zeros(length(domains1), length(domains2))

    n1 = length(domains1); n2 = length(domains2)
    l1 = similar(Vector{eltype(domains1[1].lower)}, n1)
    u1 = similar(Vector{eltype(domains1[1].upper)}, n1)
    l2 = similar(Vector{eltype(domains2[1].lower)}, n2)
    u2 = similar(Vector{eltype(domains2[1].upper)}, n2)

    @inbounds @simd for i in 1:n1
        d = domains1[i]; l1[i] = d.lower; u1[i] = d.upper
    end
    @inbounds @simd for j in 1:n2
        d = domains2[j]; l2[j] = d.lower; u2[j] = d.upper
    end

    @inbounds for j in axes(res, 2)
        lj = l2[j]; uj = u2[j]
        @inbounds for i in axes(res, 1)
            res[i, j] = radial_covfunc_two_sided_integral(
                l1[i],
                u1[i],
                lj,
                uj;
                anti_derivative = anti,
                ℓ = only(ℓ),
            )
        end
    end

    return res
end

function integrate_radial(k, domains)
    anti = radial_antiderivative(k, Val(2))
    n = length(domains)

    lowers = similar(Vector{eltype(domains[1].lower)}, n)
    uppers = similar(Vector{eltype(domains[1].upper)}, n)
    @inbounds @simd for i in 1:n
        d = domains[i]
        lowers[i] = d.lower
        uppers[i] = d.upper
    end

    res = zeros(n, n)

    # Fill upper triangle (including diagonal) in parallel
    Threads.@threads for j in 1:n
        lj = lowers[j]; uj = uppers[j]
        @inbounds for i in j:n
            li = lowers[i]; ui = uppers[i]
            res[i, j] = radial_covfunc_two_sided_integral(
                li, ui, lj, uj; anti_derivative = anti, ℓ = k.lengthscales
            )
        end
    end

    return Symmetric(res, :L)
end
