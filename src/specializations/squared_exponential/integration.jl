# Radial antiderivatives for Squared Exponential kernels
# Enables closed-form integration operations

using SpecialFunctions: erf
using KernelFunctions: SqExponentialKernel, TransformedKernel, ScaleTransform

# ============================================================================
# Lengthscale extraction for SE kernels (used by stationary integration)
# ============================================================================

_kernel_lengthscales(::SqExponentialKernel) = [1.0]
_kernel_lengthscales(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}) = [1 / _se_scale(k)]

# Constants for antiderivative formulas
const _SQRT_PI_OVER_2 = sqrt(pi / 2)
const _INV_SQRT_2 = 1 / sqrt(2)

"""
    radial_antiderivative(k::SqExponentialKernel, ::Val{1})

First radial antiderivative for one-sided integration of the SE kernel.

For the SE kernel k(r) = exp(-r²/2), the antiderivative is:
    F(r) = √(π/2) * erf(r/√2)

satisfying dF/dr = k(r).
"""
function radial_antiderivative(::SqExponentialKernel, ::Val{1})
    return r -> _SQRT_PI_OVER_2 * erf(r * _INV_SQRT_2)
end

"""
    radial_antiderivative(k::SqExponentialKernel, ::Val{2})

Second radial antiderivative for two-sided integration of the SE kernel.

For the SE kernel, the second antiderivative is:
    F₂(r) = r * √(π/2) * erf(r/√2) + exp(-r²/2) - 1

The constant -1 ensures F₂(0) = 0 for numerical stability.
satisfying dF₂/dr = F₁(r).
"""
function radial_antiderivative(::SqExponentialKernel, ::Val{2})
    return r -> r * _SQRT_PI_OVER_2 * erf(r * _INV_SQRT_2) + exp(-r^2 / 2) - 1
end

"""
    radial_antiderivative(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}, ::Val{1})

First radial antiderivative for a scaled SE kernel.

The antiderivative is in terms of normalized distance τ = r/ℓ (where r is physical distance).
The integration code handles the ℓ scaling via the `scales` parameter.
"""
function radial_antiderivative(
        ::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform},
        ::Val{1},
    )
    # Same as unit scale - integration code handles scaling
    return τ -> _SQRT_PI_OVER_2 * erf(τ * _INV_SQRT_2)
end

"""
    radial_antiderivative(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}, ::Val{2})

Second radial antiderivative for a scaled SE kernel.

The antiderivative is in terms of normalized distance τ = r/ℓ.
"""
function radial_antiderivative(
        ::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform},
        ::Val{2},
    )
    # Same as unit scale - integration code handles scaling
    return τ -> τ * _SQRT_PI_OVER_2 * erf(τ * _INV_SQRT_2) + exp(-τ^2 / 2) - 1
end
