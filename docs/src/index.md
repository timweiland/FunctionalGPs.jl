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

### GP regression with mixed observations

```julia
using FunctionalGPs
using AbstractGPs

k = WendlandKernel(1, 2, 1.0)
f = GP(k)

# Condition on function values
f1 = condition_on_observation(f, [0.0, 1.0], [0.0, 0.84]; noise = 1.0e-8)

# Condition further on a derivative observation
∂x = PartialDerivative((1,))
ℒ = EvaluationFunctional([0.5]) ∘ ∂x
f2 = condition_on_observation(f1, ℒ, [1.0]; noise = 1.0e-8)
```

### Joint functional Gaussians

For models that bundle several linear functionals of the same GP — function
values, derivatives, integrals — and need their cross-covariances preserved
(e.g. for hyperparameter inference in Turing), use [`FunctionalGaussian`](@ref):

```julia
using FunctionalGPs, FunctionalGPs.Notation

fg = FunctionalGaussian(f;
    y  = δ(X_obs),
    dy = δ(X_pred) ∘ ∂(1),
    q  = ∫([Interval(0.0, 1.0)]),
)

# Marginal log-likelihood for hyperparameter optimisation / sampling
ℓ = loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))

# Posterior over the latent (unobserved) blocks
post = posterior(fg, (; y = y_obs); noise = (; y = σ²))
post.dy   # LazyMvNormal over derivative locations
```

See the **Joint Functional Gaussians** and **Notation** pages in the API
Reference for details.

## Related Packages

- [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) - Core GP abstractions
- [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) - Kernel definitions
- [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl) - Sparse precision GPs
