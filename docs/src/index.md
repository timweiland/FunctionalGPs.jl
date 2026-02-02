```@raw html
---
layout: home

hero:
  name: "FunctionalGPs.jl"
  text: "Linear Functionals for Gaussian Processes"
  tagline: "Linear transforms of GPs. Mix & match point observations, integrals and derivatives."
  image:
    src: /logo.svg
    alt: "FunctionalGPs.jl"
  actions:
    - theme: brand
      text: "Get Started"
      link: /getting_started/
    - theme: alt
      text: "View on GitHub"
      link: https://github.com/timweiland/FunctionalGPs.jl
    - theme: alt
      text: "API Reference"
      link: /api/

features:
  - title: "🔗 Composable Functionals"
    details: "Chain evaluation, differentiation, and integration with intuitive ∘, +, ⊗ operators."
  - title: "⚡ Automatic Optimization"
    details: "Kernel trait dispatch selects the fastest algorithm - Toeplitz, sparse, or dense - automatically."
  - title: "📐 Derivative Observations"
    details: "Condition GPs on derivative data. Perfect for physics-informed learning and slope constraints."
  - title: "∫ Integral Observations"
    details: "Bayesian quadrature and cell-averaged measurements made simple with VectorizedLebesgueIntegral."
  - title: "🎯 Physics-Informed Learning"
    details: "Encode physical constraints, boundary conditions, and conservation laws directly into your GP."
  - title: "🧩 AbstractGPs Integration"
    details: "Seamlessly extends the JuliaGaussianProcesses ecosystem. Use your favorite kernels."
---
```

## What is FunctionalGPs.jl?

FunctionalGPs.jl provides a composable framework for applying linear functionals (evaluation, integration, differentiation) to Gaussian processes. Use it for GP regression with derivative or integral observations, Bayesian quadrature, physics-informed learning, and more.

## Quick Start

```julia
using Pkg
Pkg.add("FunctionalGPs")
```

```julia
using FunctionalGPs
using AbstractGPs

# GP with Wendland kernel
k = WendlandKernel(1, 2, 1.0)
f = GP(k)

# Condition on function values
obs = LinearObservation(EvaluationFunctional([0.0, 1.0]), [0.0, 0.84], 1e-8)
f1 = condition_on(f, obs)

# Condition on derivative observations
dx = PartialDerivative((1,))
L = EvaluationFunctional([0.5]) ∘ dx
obs2 = LinearObservation(L, [1.0], 1e-8)
f2 = condition_on(f1, obs2)
```

## Related Packages

- [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) - Core GP abstractions
- [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) - Kernel definitions
- [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl) - Sparse precision GPs
