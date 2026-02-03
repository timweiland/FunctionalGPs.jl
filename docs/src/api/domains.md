# Domains

The domains module provides types for representing spatial domains and grids used throughout FunctionalGPs.jl.

## Domain Types

```@docs
Domain
Interval
BoxDomain
```

## Grid Types

```@docs
FactorizedGrid
FactorizedBoxDomains
```

## Grid Construction

```@docs
uniform_grid_n
uniform_grid_step
intervals_from_endpoints
```

## Domain Operations

```@docs
volume
get_intervals
```

## Kernel Matrix Computation

Efficient kernel matrix computation for tensor product grids:

```@docs
kernelmatrix(::KernelFunctions.KernelTensorProduct, ::FactorizedGrid, ::FactorizedGrid)
kernelmatrix_diag(::KernelFunctions.KernelTensorProduct, ::FactorizedGrid)
```
