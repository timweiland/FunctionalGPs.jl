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
    radial_covfunc_one_sided_integral(a, b, x; anti_derivative = anti, ℓ = ℓ)
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
                ℓ = ℓ,
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
