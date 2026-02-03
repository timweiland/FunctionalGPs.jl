# Squared Exponential (RBF/Gaussian) kernel support
# Dispatches on KernelFunctions.SqExponentialKernel and TransformedKernel variants

using KernelFunctions:
    SqExponentialKernel, TransformedKernel, ScaleTransform, ARDTransform,
    KernelTensorProduct
using Polynomials

# ============================================================================
# Hermite polynomial utilities
# ============================================================================

"""
    probabilist_hermite(n::Int) -> Polynomial

Compute the n-th probabilist's Hermite polynomial He_n(x).

Uses the recurrence relation:
- He_0(x) = 1
- He_1(x) = x
- He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

These polynomials satisfy: d^n/dx^n[exp(-x²/2)] = (-1)^n * He_n(x) * exp(-x²/2)
"""
function probabilist_hermite(n::Int)
    n >= 0 || throw(ArgumentError("Order must be non-negative"))
    n == 0 && return Polynomial([1.0])
    n == 1 && return Polynomial([0.0, 1.0])

    He_prev = Polynomial([1.0])       # He_0
    He_curr = Polynomial([0.0, 1.0])  # He_1

    for k in 1:(n - 1)
        He_next = Polynomial([0.0, 1.0]) * He_curr - k * He_prev
        He_prev = He_curr
        He_curr = He_next
    end

    return He_curr
end

# ============================================================================
# Derivative kernel types
# ============================================================================

"""
    SEDerivativeEvenKernel{TK, TP, TC, TS} <: Kernel

Stationary kernel from an even-order derivative of the Squared Exponential kernel.

When the total derivative order (n + m) is even, the kernel remains stationary:
it depends only on |x - y|. The kernel has the form:

    k(x,y) = coefficient * He_{n+m}(s|x-y|) * exp(-s²(x-y)²/2)

where He_n is the n-th probabilist's Hermite polynomial and s is the scale factor.

# Fields
- `base`: The original SE kernel (SqExponentialKernel or TransformedKernel)
- `hermite_poly`: Probabilist's Hermite polynomial He_{n+m}
- `coefficient`: Scalar coefficient (-1)^n * s^(n+m)
- `scale`: Scale factor s (1 for base kernel, 1/ℓ for transformed)
"""
struct SEDerivativeEvenKernel{TK, TP, TC, TS} <: KernelFunctions.Kernel
    base::TK
    hermite_poly::TP
    coefficient::TC
    scale::TS
end

"""
    SEDerivativeOddKernel{TK, TP, TC, TS} <: Kernel

Signed-stationary kernel from an odd-order derivative of the Squared Exponential.

When the total derivative order (n + m) is odd, the kernel has structure
`k(x,y) = sign(x-y) * f(|x-y|)`. The kernel has the form:

    k(x,y) = sign(x-y) * coefficient * |He_{n+m}(s|x-y|)| * exp(-s²(x-y)²/2)

where He_n is the n-th probabilist's Hermite polynomial and s is the scale factor.

# Fields
- `base`: The original SE kernel
- `hermite_poly`: Probabilist's Hermite polynomial He_{n+m}
- `coefficient`: Scalar coefficient (-1)^n * s^(n+m)
- `scale`: Scale factor s
"""
struct SEDerivativeOddKernel{TK, TP, TC, TS} <: KernelFunctions.Kernel
    base::TK
    hermite_poly::TP
    coefficient::TC
    scale::TS
end

# Trait registrations
kernel_structure(::SqExponentialKernel) = StationaryKernelTrait()
kernel_structure(::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}) = StationaryKernelTrait()
kernel_structure(::SEDerivativeEvenKernel) = StationaryKernelTrait()
kernel_structure(::SEDerivativeOddKernel) = SignedStationaryKernelTrait()

# ============================================================================
# Scale extraction utilities
# ============================================================================

"""
    _se_scale(k::SqExponentialKernel) -> Float64

Extract the scale factor from an SE kernel. Returns 1.0 for the base kernel.
"""
_se_scale(::SqExponentialKernel) = 1.0

"""
    _se_scale(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}) -> Float64

Extract the scale factor from a transformed SE kernel.
"""
function _se_scale(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform})
    return only(k.transform.s)
end

# ============================================================================
# Kernel evaluation
# ============================================================================

function (k::SEDerivativeEvenKernel)(x, y)
    diff = only(x) - only(y)
    τ = k.scale * diff
    return k.coefficient * k.hermite_poly(τ) * exp(-τ^2 / 2)
end

function (k::SEDerivativeOddKernel)(x, y)
    diff = only(x) - only(y)
    iszero(diff) && return zero(diff)
    τ = k.scale * diff
    # For odd Hermite polynomials, He_n(τ) = sign(τ) * He_n(|τ|)
    # So He_n(τ) = sign(diff) * He_n(|τ|) since τ = s * diff and s > 0
    return k.coefficient * k.hermite_poly(τ) * exp(-τ^2 / 2)
end

# ============================================================================
# Derivative computation
# ============================================================================

"""
    derivative(k::SqExponentialKernel, n::Int, m::Int)

Compute the mixed partial derivative kernel ∂^(n+m)k/∂x^n∂y^m for a 1D SE kernel.

The derivative has the form:
    ∂^(n+m)k/∂x^n∂y^m = (-1)^n * He_{n+m}(x-y) * exp(-(x-y)²/2)

where He_n is the n-th probabilist's Hermite polynomial.

Returns a `DerivativeKernel1D` wrapping either an even or odd derivative kernel.
"""
function derivative(k::SqExponentialKernel, n::Int, m::Int)
    n >= 0 || throw(ArgumentError("Derivative order n must be non-negative"))
    m >= 0 || throw(ArgumentError("Derivative order m must be non-negative"))

    total = n + m
    total == 0 && return k

    He = probabilist_hermite(total)
    s = _se_scale(k)
    coeff = (-1)^n * s^total

    inner = if iseven(total)
        SEDerivativeEvenKernel(k, He, coeff, s)
    else
        SEDerivativeOddKernel(k, He, coeff, s)
    end

    return DerivativeKernel1D{n, m}(k, inner)
end

"""
    derivative(k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform}, n::Int, m::Int)

Compute the mixed partial derivative for a scaled SE kernel.

For k_t(x,y) = k(sx, sy) = exp(-s²(x-y)²/2), the derivative is:
    ∂^(n+m)k_t/∂x^n∂y^m = (-1)^n * s^(n+m) * He_{n+m}(s(x-y)) * exp(-s²(x-y)²/2)
"""
function derivative(
        k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform},
        n::Int,
        m::Int,
    )
    n >= 0 || throw(ArgumentError("Derivative order n must be non-negative"))
    m >= 0 || throw(ArgumentError("Derivative order m must be non-negative"))

    total = n + m
    total == 0 && return k

    He = probabilist_hermite(total)
    s = _se_scale(k)
    coeff = (-1)^n * s^total

    inner = if iseven(total)
        SEDerivativeEvenKernel(k, He, coeff, s)
    else
        SEDerivativeOddKernel(k, He, coeff, s)
    end

    return DerivativeKernel1D{n, m}(k, inner)
end

# ============================================================================
# Multi-dimensional support via tensor product decomposition
# ============================================================================

# The SE kernel has an inherent tensor product structure:
#   k(x,y) = exp(-Σᵢ (xᵢ-yᵢ)²/(2ℓᵢ²)) = Πᵢ exp(-(xᵢ-yᵢ)²/(2ℓᵢ²))
#
# This allows multi-D derivatives to be computed as tensor products of 1D derivatives.

"""
    se_tensor_product(ndims::Int)
    se_tensor_product(scales::AbstractVector)

Decompose an SE kernel into a `KernelTensorProduct` of 1D SE kernels.

# Arguments
- `ndims`: Number of dimensions (uses unit lengthscale in each dimension)
- `scales`: Per-dimension scale factors (s = 1/ℓ)

# Examples
```julia
# 2D SE with unit lengthscales
k2d = se_tensor_product(2)

# 2D SE with different lengthscales
k2d_ard = se_tensor_product([1/0.5, 1/1.0])  # ℓ₁=0.5, ℓ₂=1.0

# Apply partial derivatives
∂xy = PartialDerivative((1, 1))
dk = ∂xy(k2d_ard)  # Returns KernelTensorProduct of 1D derivatives
```
"""
function se_tensor_product(ndims::Int)
    ndims > 0 || throw(ArgumentError("ndims must be positive"))
    return KernelTensorProduct([SqExponentialKernel() for _ in 1:ndims])
end

function se_tensor_product(scales::AbstractVector)
    isempty(scales) && throw(ArgumentError("scales must not be empty"))
    kernels = [SqExponentialKernel() ∘ ScaleTransform(s) for s in scales]
    return KernelTensorProduct(kernels)
end

"""
    _se_scales(k::TransformedKernel{<:SqExponentialKernel, <:ARDTransform})

Extract per-dimension scale factors from an ARD-transformed SE kernel.
"""
function _se_scales(k::TransformedKernel{<:SqExponentialKernel, <:ARDTransform})
    return k.transform.v
end

"""
    _to_tensor_product(k::SqExponentialKernel, ndims::Int)

Convert an SE kernel to tensor product form for multi-D derivative computation.
"""
_to_tensor_product(::SqExponentialKernel, ndims::Int) = se_tensor_product(ndims)

function _to_tensor_product(
        k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform},
        ndims::Int,
    )
    s = _se_scale(k)
    return se_tensor_product(fill(s, ndims))
end

function _to_tensor_product(k::TransformedKernel{<:SqExponentialKernel, <:ARDTransform}, ::Int)
    scales = _se_scales(k)
    return se_tensor_product(scales)
end
