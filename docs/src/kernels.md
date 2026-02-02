# Kernels

FunctionalGPs.jl provides specialized kernel implementations optimized for linear functional operations (evaluation, differentiation, integration).

## Overview

The kernels module offers:
- **Matérn kernels** with half-integer smoothness and analytic derivatives
- **Wendland kernels** with compact support for sparse computations
- **Kernel trait system** for automatic algorithm selection

All kernels integrate with [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) and can be used anywhere a `Kernel` is expected.

## Matérn Kernels

The Matérn family provides tunable smoothness via the parameter ν. Half-integer values (ν = 1/2, 3/2, 5/2, ...) admit closed-form expressions.

```@docs
HalfIntegerMaternKernel
```

### Derivative Kernels

Derivatives of Matérn kernels are computed analytically:

```@docs
derivative
DerivativeKernel1D
```

## Wendland Kernels

Wendland kernels are compactly supported (exactly zero beyond a radius), enabling sparse kernel matrices for large-scale problems.

```@docs
WendlandKernel
WendlandPolynomial
```

## Compact Kernels

The compact kernel hierarchy provides the foundation for Wendland and other compactly-supported kernels.

### Abstract Types

```@docs
AbstractCompactKernel
AbstractCompactRadialKernel
AbstractCompactSignedRadialKernel
```

### Concrete Types

```@docs
CompactPolynomialKernel
CompactSignedPolynomialKernel
```

## Kernel Traits

The trait system enables automatic dispatch to optimized algorithms (Toeplitz, sparse, etc.) based on kernel structure.

```@docs
KernelStructureTrait
GenericKernelTrait
StationaryKernelTrait
SignedStationaryKernelTrait
kernel_structure
```

## Covariance Assembly Functions

These functions assemble covariance matrices from kernels and functionals, dispatching on kernel traits for optimal performance.

```@docs
kernel_evaluate_evaluate
kernel_integrate_integrate
kernel_integrate_evaluate
```
