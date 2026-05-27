export LinearlyScaledKernel

"""
    LinearlyScaledKernel{K, S} <: Kernel

A kernel scaled by an arbitrary (possibly negative) factor.

Unlike `KernelFunctions.ScaledKernel`, this allows negative scaling, which arises
when linear function operators with negative coefficients are applied to kernels.
The result is not a valid covariance kernel on its own, but participates correctly
in cross-covariance computations via linear combinations.

# Fields
- `kernel::K`: The underlying kernel
- `scalar::S`: The scaling factor (may be negative; stored like ScaledKernel's σ²)
"""
struct LinearlyScaledKernel{K <: Kernel, S} <: Kernel
    kernel::K
    scalar::S
end

(k::LinearlyScaledKernel)(x, y) = first(k.scalar) * k.kernel(x, y)

function kernelmatrix(k::LinearlyScaledKernel, x::AbstractVector)
    return first(k.scalar) * kernelmatrix(k.kernel, x)
end

function kernelmatrix(k::LinearlyScaledKernel, x::AbstractVector, y::AbstractVector)
    return first(k.scalar) * kernelmatrix(k.kernel, x, y)
end

function kernelmatrix_diag(k::LinearlyScaledKernel, x::AbstractVector)
    return first(k.scalar) * kernelmatrix_diag(k.kernel, x)
end
