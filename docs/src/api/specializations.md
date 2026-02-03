# Specializations

The specializations module provides trait-dispatched implementations for specific kernel types, enabling optimized matrix representations and operations.

## Overview

FunctionalGPs.jl uses a trait system to automatically select the most efficient algorithm for assembling covariance matrices. When you apply functionals (evaluation, integration, differentiation) to a kernel, the system:

1. Queries the kernel's structure via `kernel_structure(k)`
2. Dispatches to a specialized implementation based on the trait
3. Returns a lazy or sparse matrix representation when available

This happens transparently - you get optimal performance without manual intervention.

## Kernel Structure Traits

```@docs
KernelStructureTrait
GenericKernelTrait
StationaryKernelTrait
SignedStationaryKernelTrait
kernel_structure
```

## Trait-Dispatched Operations

These functions form the core interface for efficient kernel operations:

```@docs
kernel_evaluate_evaluate
kernel_integrate_integrate
kernel_integrate_evaluate
```

## Lazy Matrix Types

Specialized kernels return lazy matrix representations that defer computation:

```@docs
StationaryKernelMatrix
SignedStationaryKernelMatrix
```

## Stationary Kernel Specifications

For kernels that are stationary (depend only on `|x - y|`), the system uses compact specifications:

```@docs
StationaryKernelSpec
SignedStationaryKernelSpec
stationary_kernel_spec
radial_antiderivative
```

## Supported Kernel Specializations

### Half-Integer Matern Kernels

Half-integer Matern kernels (`HalfIntegerMaternKernel{P}` for smoothness `nu = P + 1/2`) support:
- Lazy evaluation via `StationaryKernelMatrix`
- Toeplitz matrices for uniformly spaced points
- Closed-form radial antiderivatives for integration
- Automatic derivative kernel construction

```julia
using FunctionalGPs

# Create a Matern 5/2 kernel
k = HalfIntegerMaternKernel(2, [1.0])

# The trait system recognizes this as stationary
kernel_structure(k)  # Returns StationaryKernelTrait()

# Evaluation returns a lazy matrix
X = collect(range(0, 1, length=100))
K = kernel_evaluate_evaluate(k, X)  # StationaryKernelMatrix

# Derivatives are also specialized
dk = derivative(k, 1, 0)  # First derivative kernel
kernel_structure(dk.inner_kernel)  # SignedStationaryKernelTrait()
```

### Compact Polynomial Kernels

Compact kernels (including Wendland kernels) have finite support and return sparse matrices:

```julia
using FunctionalGPs, SparseArrays

# Wendland kernel with support radius 1.0
k = WendlandKernel(1, 2, 1.0)

# Kernel matrix is sparse
X = collect(range(0, 2, length=100))
K = kernelmatrix(k, X)
issparse(K)  # true

# Integration also produces sparse matrices
domains = [Interval(i*0.1, (i+1)*0.1) for i in 0:19]
K_int = kernel_integrate_integrate(StationaryKernelTrait(), k.inner_kernel, domains)
```

## Extending with Custom Kernels

To add specialization support for a custom kernel:

1. **Declare the trait** by implementing `kernel_structure`:
```julia
kernel_structure(::MyKernel) = StationaryKernelTrait()
```

2. **Provide a stationary specification** if applicable:
```julia
function stationary_kernel_spec(k::MyKernel, ::Type{T}) where {T}
    scales = T[1 / k.lengthscale]
    radial_map = r2 -> my_radial_function(sqrt(r2))
    return StationaryKernelSpec(scales, radial_map)
end
```

3. **Implement radial antiderivatives** for integration support:
```julia
function radial_antiderivative(k::MyKernel, ::Val{1})
    return r -> my_first_antiderivative(r)
end

function radial_antiderivative(k::MyKernel, ::Val{2})
    return r -> my_second_antiderivative(r)
end
```
