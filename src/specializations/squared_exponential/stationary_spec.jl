# Stationary kernel specifications for Squared Exponential kernels
# Enables efficient lazy matrix construction

using KernelFunctions: SqExponentialKernel, TransformedKernel, ScaleTransform

"""
    stationary_kernel_spec(k::SqExponentialKernel, ::Type{T})

Return a `StationaryKernelSpec` for the base Squared Exponential kernel (unit scale).
The radial map evaluates exp(-r²/2) at distances.
"""
function stationary_kernel_spec(::SqExponentialKernel, ::Type{T}) where {T <: Real}
    scales = [one(T)]
    radial_map = r2 -> exp(-max(r2, zero(r2)) / 2)
    return StationaryKernelSpec(scales, radial_map)
end

"""
    stationary_kernel_spec(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}, ::Type{T})

Return a `StationaryKernelSpec` for a scaled SE kernel.
The scale factor s = 1/ℓ is incorporated into the scales vector.
"""
function stationary_kernel_spec(
        k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform},
        ::Type{T},
    ) where {T <: Real}
    # Don't convert scale to T — it may carry AD dual numbers
    s = _se_scale(k)
    scales = [s]
    radial_map = r2 -> exp(-max(r2, zero(r2)) / 2)
    return StationaryKernelSpec(scales, radial_map)
end

"""
    stationary_kernel_spec(k::SEDerivativeEvenKernel, ::Type{T})

Return a `StationaryKernelSpec` for even-order SE derivatives.
"""
function stationary_kernel_spec(
        k::SEDerivativeEvenKernel,
        ::Type{T},
    ) where {T <: Real}
    # Don't convert to T — these may carry AD dual numbers
    s = k.scale
    scales = [s]

    radial_map = let kernel = k, coeff = k.coefficient
        r2 -> begin
            τ = _safe_dist(r2)
            return coeff * kernel.hermite_poly(τ) * exp(-τ^2 / 2)
        end
    end

    return StationaryKernelSpec(scales, radial_map)
end

"""
    stationary_kernel_spec(k::SEDerivativeOddKernel, ::Type{T})

Return a `SignedStationaryKernelSpec` for odd-order SE derivatives.
"""
function stationary_kernel_spec(
        k::SEDerivativeOddKernel,
        ::Type{T},
    ) where {T <: Real}
    # Don't convert to T — these may carry AD dual numbers
    s = k.scale
    scales = [s]

    signed_map = let kernel = k, coeff = k.coefficient
        (r2, sign_val) -> begin
            τ = _safe_dist(r2)
            return sign_val * coeff * kernel.hermite_poly(τ) * exp(-τ^2 / 2)
        end
    end

    return SignedStationaryKernelSpec(scales, signed_map)
end
