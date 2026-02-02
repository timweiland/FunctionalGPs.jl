# VectorizedLebesgueIntegral applied to Kernels → creates IntegralPVCrosscov

import KernelFunctions: KernelTensorProduct

# Generic construction - works for any kernel with 1D interval domains
function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
        k::Kernel;
        arg = 2,
    ) where {T}
    return IntegralPVCrosscov(k, ℒ.domains, arg)
end

# Disambiguation for ScaledKernel - scale the result
function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
        k::ScaledKernel;
        arg = 2,
    ) where {T}
    return k.σ² * ℒ(k.kernel; arg = arg)
end

function cancel_integral(
        k::DerivativeKernel1D{N, M},
        ℒ::VectorizedLebesgueIntegral{Interval{T}};
        arg = 2,
        same_arg = true,
    ) where {T, N, M}
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

# Main entry point - delegates to trait-based helper
function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(
        k::DerivativeKernel1D{N, M};
        arg = 2,
    ) where {T, N, M}
    return _apply_integral_to_derivative_kernel(
        kernel_structure(k.original_kernel), k, ℒ; arg = arg
    )
end

# Stationary kernels: can always cancel, even for "wrong argument" case
# This exploits the property ∂k/∂x = -∂k/∂y for stationary kernels k(x,y) = φ(x-y)
function _apply_integral_to_derivative_kernel(
        ::StationaryKernelTrait,
        k::DerivativeKernel1D{N, M},
        ℒ::VectorizedLebesgueIntegral{Interval{T}};
        arg = 2,
    ) where {T, N, M}
    if ((N == 0) && (arg == 1)) || ((M == 0) && (arg == 2))
        return -cancel_integral(k, ℒ; arg = arg, same_arg = false)
    end
    return cancel_integral(k, ℒ; arg = arg)
end

# Non-stationary kernels: error for "wrong argument" case
function _apply_integral_to_derivative_kernel(
        ::KernelStructureTrait,
        k::DerivativeKernel1D{N, M},
        ℒ::VectorizedLebesgueIntegral{Interval{T}};
        arg = 2,
    ) where {T, N, M}
    if ((N == 0) && (arg == 1)) || ((M == 0) && (arg == 2))
        error("Integration of derivative kernel on non-matching argument not implemented for non-stationary kernels")
    end
    return cancel_integral(k, ℒ; arg = arg)
end

box_integrals(x, y) = throw(MethodError(box_integrals, (x, y)))

function box_integrals(k::KernelTensorProduct, domains::FactorizedBoxDomains; arg = 2)
    if length(k.kernels) != ndims(domains)
        throw(
            ArgumentError(
                "Number of kernels $(length(k.kernels)) must match number of domains $(length(domains))",
            ),
        )
    end
    ℒs = map(VectorizedLebesgueIntegral, get_intervals(domains))
    return mapreduce(
        args -> ((cur_k, cur_ℒ) = args; cur_ℒ(cur_k; arg = arg)),
        ⊗,
        zip(k.kernels, ℒs),
    )
end

function (ℒ::VectorizedLebesgueIntegral{BoxDomain{T}})(
        k::KernelTensorProduct;
        arg = 2,
    ) where {T}
    return box_integrals(k, ℒ.domains; arg = arg)
end
