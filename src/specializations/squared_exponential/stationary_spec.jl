# Stationary kernel specifications for Squared Exponential kernels
# Enables efficient lazy matrix construction

using KernelFunctions: SqExponentialKernel, TransformedKernel, ScaleTransform

"""
    stationary_kernel_spec(k::SqExponentialKernel, ::Type{T})

Return a `StationaryKernelSpec` for the base Squared Exponential kernel (unit scale).
The radial map evaluates exp(-r²/2) at distances.
"""
function stationary_kernel_spec(::SqExponentialKernel, ::Type{T}) where {T <: AbstractFloat}
    scales = T[one(T)]  # Unit scale
    radial_map = let zero_T = zero(T)
        r2 -> exp(-max(r2, zero_T) / 2)
    end
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
    ) where {T <: AbstractFloat}
    s = T(_se_scale(k))
    scales = T[s]
    radial_map = let zero_T = zero(T)
        r2 -> exp(-max(r2, zero_T) / 2)
    end
    return StationaryKernelSpec(scales, radial_map)
end

"""
    stationary_kernel_spec(k::SEDerivativeEvenKernel, ::Type{T})

Return a `StationaryKernelSpec` for even-order SE derivatives.
"""
function stationary_kernel_spec(
        k::SEDerivativeEvenKernel,
        ::Type{T},
    ) where {T <: AbstractFloat}
    s = T(k.scale)
    scales = T[s]
    coeff_T = T(k.coefficient)

    radial_map = let kernel = k, coeff = coeff_T, zero_T = zero(T)
        r2 -> begin
            τ = sqrt(max(r2, zero_T))
            return coeff * kernel.hermite_poly(T(τ)) * exp(-τ^2 / 2)
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
    ) where {T <: AbstractFloat}
    s = T(k.scale)
    scales = T[s]
    coeff_T = T(k.coefficient)

    signed_map = let kernel = k, coeff = coeff_T, zero_T = zero(T)
        (r2, sign_val) -> begin
            τ = sqrt(max(r2, zero_T))
            # For odd Hermite: He_n(-τ) = -He_n(τ)
            # So we evaluate at |τ| and multiply by sign
            return sign_val * coeff * kernel.hermite_poly(T(τ)) * exp(-τ^2 / 2)
        end
    end

    return SignedStationaryKernelSpec(scales, signed_map)
end
