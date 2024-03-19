export VectorizedLebesgueIntegral

struct VectorizedLebesgueIntegral{T<:Domain} <: AbstractLinearFunctional
    domains::AbstractArray{T}

    function VectorizedLebesgueIntegral(domains::AbstractArray{T}) where {T}
        if length(domains) == 0
            throw(ArgumentError("At least one domain must be provided"))
        end
        return new{T}(domains)
    end

    VectorizedLebesgueIntegral(domains...) = VectorizedLebesgueIntegral(domains)
    end

output_shape(ℒ::VectorizedLebesgueIntegral) = size(ℒ.domains)

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
    k::CompactPolynomialKernel;
    arg = 2,
) where {T}
    return CompactPolynomialCovFunc1D_Identity_LebesgueIntegral(k, ℒ.domains, arg)
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
    pv::CompactPolynomialCovFunc1D_Identity_LebesgueIntegral,
) where {T}
    return integrate(pv.covfunc, ℒ.domains, pv.domains)
end

function cancel_integral(
    k::DerivativeKernel1D{N,M},
    ℒ::VectorizedLebesgueIntegral{Interval{T}};
    arg = 2,
    same_arg = true,
) where {T,N,M}
    # @assert (arg == 2) ? M > 0 : N > 0
    eval_b = EvaluationFunctional(map(d -> d.upper, ℒ.domains))
    eval_a = EvaluationFunctional(map(d -> d.lower, ℒ.domains))
    if arg == 2
        k_integrated =
            same_arg ? derivative(k.original_kernel, N, M - 1) :
            derivative(k.original_kernel, N - 1, M)
        return eval_b(k_integrated) - eval_a(k_integrated)
    else
        k_integrated =
            same_arg ? derivative(k.original_kernel, N - 1, M) :
            derivative(k.original_kernel, N, M - 1)
        return eval_b(k_integrated; arg = arg) - eval_a(k_integrated; arg = arg)
    end
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
    k::DerivativeKernel1D{N,M};
    arg = 2,
) where {T,N,M}
    return cancel_integral(k, ℒ; arg = arg)
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
    k::DerivativeKernel1D{N,0,<:AbstractCompactRadialKernel};
    arg = 2,
) where {T,N}
    if arg == 1
        return cancel_integral(k, ℒ; arg = arg)
    else
        return -cancel_integral(k, ℒ; arg = arg, same_arg = false)
    end
end

function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
    k::DerivativeKernel1D{0,M,<:AbstractCompactRadialKernel};
    arg = 2,
) where {T,M}
    if arg == 2
        return cancel_integral(k, ℒ; arg = arg)
    else
        return -cancel_integral(k, ℒ; arg = arg, same_arg = false)
    end
end

function (op::PartialDerivative{1, 1})(pv::CompactPolynomialCovFunc1D_Identity_LebesgueIntegral)
    k = pv.covfunc
    dk = op(k, arg=randproc_arg(pv))
    ℒ = VectorizedLebesgueIntegral(pv.domains)
    return ℒ(dk, arg=randvar_arg(pv))
end
