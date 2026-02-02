using KernelFunctions: Kernel
import KernelFunctions: kernelmatrix, kernelmatrix_diag

export DerivativeKernel1D, derivative

"""
    DerivativeKernel1D{N, M, K, KD} <: Kernel

A kernel representing the `N`-th and `M`-th mixed partial derivative of a 1D kernel.

Given an original kernel `k(x, y)`, this type represents
`∂ᴺ⁺ᴹk/∂xᴺ∂yᴹ`. The struct wraps both the original kernel and the
pre-computed derivative kernel for efficient evaluation.

# Type Parameters
- `N`: Derivative order with respect to the first argument
- `M`: Derivative order with respect to the second argument
- `K`: Type of the original kernel
- `KD`: Type of the derivative kernel

# Fields
- `original_kernel`: The base kernel from which derivatives are taken
- `derivative_kernel`: The computed derivative kernel used for evaluation
- `order1`: Derivative order `N` (first argument)
- `order2`: Derivative order `M` (second argument)

# Examples
```julia
k = HalfIntegerMaternKernel(2, [1.0])
dk = derivative(k, 1, 1)  # Returns DerivativeKernel1D{1,1,...}
dk([0.3], [0.7])  # Evaluate ∂²k/∂x∂y at (0.3, 0.7)
```

See also: [`derivative`](@ref), [`HalfIntegerMaternKernel`](@ref)
"""
struct DerivativeKernel1D{N, M, K <: Kernel, KD <: Kernel} <: Kernel
    original_kernel::K
    derivative_kernel::KD
    order1::Int
    order2::Int

    function DerivativeKernel1D{N, M}(kernel::K, derivative_kernel::KD) where {N, M, K, KD}
        @assert N isa Int && M isa Int
        return new{N, M, K, KD}(kernel, derivative_kernel, N, M)
    end
end

(k::DerivativeKernel1D)(x, y) = k.derivative_kernel(x, y)

# DerivativeKernel1D inherits all behavior from the internal kernel
function kernelmatrix(k::DerivativeKernel1D, x::AbstractVector, y::AbstractVector)
    return kernelmatrix(k.derivative_kernel, x, y)
end
function kernelmatrix(k::DerivativeKernel1D, x::AbstractVector)
    return kernelmatrix(k.derivative_kernel, x)
end
function kernelmatrix_diag(k::DerivativeKernel1D, x::AbstractVector)
    return kernelmatrix_diag(k.derivative_kernel, x)
end

"""
    derivative(k::Kernel, n::Int, m::Int)

Compute the mixed partial derivative kernel `∂ⁿ⁺ᵐk/∂xⁿ∂yᵐ`.

Returns a [`DerivativeKernel1D`](@ref) representing the `n`-th partial
derivative with respect to the first argument and `m`-th partial derivative
with respect to the second argument.

# Arguments
- `k`: The kernel to differentiate (must support derivatives)
- `n`: Derivative order with respect to the first argument (≥ 0)
- `m`: Derivative order with respect to the second argument (≥ 0)

# Returns
A `DerivativeKernel1D{n, m, ...}` that can be evaluated like any kernel.

# Examples
```julia
k = HalfIntegerMaternKernel(2, [1.0])

# First derivative in both arguments: ∂²k/∂x∂y
k11 = derivative(k, 1, 1)

# Second derivative in x only: ∂²k/∂x²
k20 = derivative(k, 2, 0)

# Chained derivatives accumulate orders
k21 = derivative(k11, 1, 0)  # Equivalent to derivative(k, 2, 1)
```

# Supported Kernels
- [`HalfIntegerMaternKernel`](@ref): Analytic derivatives via polynomial recurrence
- [`CompactPolynomialKernel`](@ref): Derivatives of Wendland and compact kernels

See also: [`DerivativeKernel1D`](@ref)
"""
function derivative(k::DerivativeKernel1D{N1, M1}, N2::Int, M2::Int) where {N1, M1}
    if N2 == 0 && M2 == 0
        return k
    end
    return derivative(k.original_kernel, N1 + N2, M1 + M2)
end

function Base.:(==)(k1::DerivativeKernel1D, k2::DerivativeKernel1D)
    return k1.original_kernel == k2.original_kernel &&
        k1.derivative_kernel == k2.derivative_kernel
end

function Base.isapprox(k1::DerivativeKernel1D, k2::DerivativeKernel1D)
    return k1.original_kernel ≈ k2.original_kernel &&
        k1.derivative_kernel ≈ k2.derivative_kernel
end

function kernel_evaluate_evaluate(k::DerivativeKernel1D, X)
    return kernel_evaluate_evaluate(k.derivative_kernel, X)
end

function kernel_evaluate_evaluate(k::DerivativeKernel1D, X1, X2)
    return kernel_evaluate_evaluate(k.derivative_kernel, X1, X2)
end
