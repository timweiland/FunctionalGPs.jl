using KernelFunctions: Kernel
import KernelFunctions: kernelmatrix, kernelmatrix_diag

export DerivativeKernel1D, derivative

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

function derivative(k::DerivativeKernel1D{N1, M1}, N2::Int, M2::Int) where {N1, M1}
    if N2 == 0 && M2 == 0
        return k
    end
    return DerivativeKernel1D{N1 + N2, M1 + M2}(
        k.original_kernel,
        derivative(k.derivative_kernel, N2, M2).derivative_kernel,
    )
end

function Base.:(==)(k1::DerivativeKernel1D, k2::DerivativeKernel1D)
    return k1.original_kernel == k2.original_kernel &&
        k1.derivative_kernel == k2.derivative_kernel
end

function Base.isapprox(k1::DerivativeKernel1D, k2::DerivativeKernel1D)
    return k1.original_kernel ≈ k2.original_kernel &&
        k1.derivative_kernel ≈ k2.derivative_kernel
end
