export RadialCovarianceFunction1D_Identity_LebesgueIntegral,
    radial_covfunc_one_sided_integral, radial_covfunc_two_sided_integral

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
    anti = radial_antiderivative(covfunc(pv), 1)
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
