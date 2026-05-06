# GP Conditioning

This module provides tools for conditioning Gaussian processes on observations
made through linear functionals. This enables GP regression with not just point
observations, but also derivatives, integrals, and other linear transformations.

## Overview

The core workflow is:

1. Create a GP prior with a kernel
2. Define observations using `LinearObservation`
3. Condition the GP using `condition_on_observation`
4. Query the posterior with `mean`, `var`, `cov`, or `rand`

```julia
using FunctionalGPs, AbstractGPs

# Prior GP
k = Matern52Kernel()
f = GP(k)

# Observe function values
X = [0.0, 0.5, 1.0]
y = sin.(X)
f_post = condition_on_observation(f, X, y; noise=0.01)

# Posterior predictions
X_new = 0:0.1:1
posterior_mean = mean(f_post(X_new))
posterior_var = var(f_post(X_new))
```

## Types

```@docs
LinearObservation
LinearConditionalGP
```

## Conditioning

```@docs
condition_on_observation
```

## Posterior Access

After conditioning, you can access properties of the posterior:

```@docs
G_chol
representer_weights
predictive_residual
y_vec
```

## Noise Access

```@docs
FunctionalGPs.εs
FunctionalGPs.μs
```

## Applying Functionals to GPs

Linear functionals can be applied directly to GPs (both prior and posterior)
to obtain the distribution of the transformed quantity:

```julia
# Prior GP
f = GP(Matern52Kernel())

# Distribution of integral over [0, 1]
ℒ = VectorizedLebesgueIntegral(Interval(0.0, 1.0))
integral_dist = ℒ(f)  # Returns MvNormal

# For a posterior GP
f_post = condition_on_observation(f, [0.5], [0.0]; noise=0.01)
posterior_integral = ℒ(f_post; noise=1e-6)
```

This works because linear functionals preserve Gaussianity: if `f ~ GP(m, k)`,
then `ℒ(f) ~ Normal(ℒ(m), ℒ(ℒ(k)))`.

## See also

For models that need to bundle several linear functionals of the same GP into
a single joint Gaussian — preserving the cross-covariances between blocks
that are otherwise dropped by independent per-functional treatment, e.g.
inside a Turing or DynamicPPL model — see the **Joint Functional Gaussians**
page in this API Reference.
