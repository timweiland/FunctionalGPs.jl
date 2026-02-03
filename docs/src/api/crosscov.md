# Cross-Covariance Reference

The cross-covariance module provides types for representing the intermediate result
of applying a linear functional to one argument of a kernel. These "PV crosscovs"
(process-vector cross-covariances) can then be composed with additional functionals
to produce covariance matrices.

## Overview

When a linear functional is applied to a kernel, the result is a
[`ProcessVectorCrossCovariance`](@ref). This represents the covariance between:
- A finite-dimensional random vector (from the functional)
- The remaining Gaussian process

```julia
# Workflow
pv = functional(kernel)      # → ProcessVectorCrossCovariance
K = functional2(pv)          # → covariance matrix
```

## Base Types and Functions

```@docs
ProcessVectorCrossCovariance
randvar_batch_size
randvar_length
randvar_arg
randproc_arg
kernelmatrix(::ProcessVectorCrossCovariance, ::AbstractVector)
```

## Concrete Crosscov Types

### Evaluation

```@docs
EvaluationPVCrosscov
```

### Integration

```@docs
IntegralPVCrosscov
```

### Stacking

```@docs
StackedPVCrosscov
```

## Arithmetic Operations

Crosscovs support algebraic composition to build complex covariance structures.

### Scaling

```@docs
AbstractScaledPVCrosscov
ConstantScaledPVCrosscov
scale
pv_crosscov
```

### Addition

```@docs
AbstractSumPVCrosscov
SumPVCrosscov
summands
```

### Tensor Products

```@docs
AbstractTensorProductCrosscov
TensorProductCrosscov
factors
⊗(::ProcessVectorCrossCovariance, ::ProcessVectorCrossCovariance)
```

## Usage Examples

### Basic Evaluation

```julia
using FunctionalGPs
using KernelFunctions

# Create kernel and evaluation points
k = SqExponentialKernel()
X = [0.0, 0.5, 1.0]

# Apply evaluation functional to kernel
δ = EvaluationFunctional(X)
pv = δ(k)  # EvaluationPVCrosscov

# Compute covariance matrix by evaluating at more points
Y = [0.25, 0.75]
K = kernelmatrix(pv, Y)  # 3×2 matrix
```

### Combining Crosscovs

```julia
# Scaling
scaled_pv = 2.0 * pv

# Addition (same evaluation points required)
pv1 = EvaluationFunctional(X)(k1)
pv2 = EvaluationFunctional(X)(k2)
sum_pv = pv1 + pv2

# Tensor products for multi-dimensional grids
pv_x = EvaluationFunctional(X_x)(k)
pv_y = EvaluationFunctional(X_y)(k)
tensor_pv = pv_x ⊗ pv_y
```

### Integration

```julia
using DomainSets

# Create integral functional
domains = [Interval(0.0, 0.5), Interval(0.5, 1.0)]
ℒ = VectorizedLebesgueIntegral(domains)

# Apply to kernel
k = HalfIntegerMaternKernel(1, [0.8])
pv = ℒ(k)  # IntegralPVCrosscov

# Get integral-evaluation covariance
X = [0.25, 0.75]
K = EvaluationFunctional(X)(pv)
```
