# Getting Started

Welcome! This page walks you through FunctionalGPs.jl from a fresh REPL to a
GP posterior conditioned on a mix of point, derivative, and integral
observations — in about five minutes.

## Why FunctionalGPs.jl?

Real measurements rarely come as clean point values of the latent function:

- A weather station reports an **integral**: average temperature over the
  past hour, not the instantaneous value.
- A wind sensor reports a **derivative**: the gradient of pressure, not its
  height.
- A satellite tile reports a **cell average** over a box: the integral of
  reflectance over the pixel footprint.
- A physical model imposes **conservation laws** — equations on derivatives
  of the latent field that hold at every point in the domain.

All of these are **linear functionals** of the unknown function. Gaussian
processes are closed under linear functionals: if `f ~ GP(m, k)` then `ℒ(f)`
is multivariate Gaussian for any linear `ℒ`, and you can write down the
cross-covariance between any two functionals in closed form.

FunctionalGPs.jl makes this composable. You build functionals out of three
primitives — point evaluation `δ`, partial differentiation `∂`, and Lebesgue
integration `∫` — and combine them with three operators:

- `∘` composes a functional with a differential operator (`δ ∘ ∂x`),
- `+` and `-` linearly combine functionals,
- `⊗` builds tensor products for axis-separable domains.

The library handles the cross-covariance assembly, the matrix algebra, and
— under the hood — dispatches to specialised paths (Toeplitz matrices for
stationary kernels on regular grids, sparse matrices for compactly supported
kernels, closed-form antiderivatives for half-integer Matérns) so you don't
have to hand-roll anything for the common cases.

## Installation

FunctionalGPs.jl is not yet in the Julia General registry. Install from
GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/timweiland/FunctionalGPs.jl")
```

This is an **alpha** release: the API will change. Pin a specific version
if you need reproducibility.

## Your first mixed-observation GP

Suppose the latent function is `f(x) = sin(2πx)` on `[0, 1]`, and you have
three kinds of observations:

- **Point values** of `f` at a handful of locations,
- **Derivative values** `f'(x)` at a couple of locations,
- **Integral values** `∫_a^b f(x) dx` over a few sub-intervals.

You don't know which kind of observation is which "more informative" — the
right thing is to condition on all of them jointly.

```julia
using FunctionalGPs
using AbstractGPs
using Random

f_true(x)  = sin(2π * x)
f_deriv(x) = 2π * cos(2π * x)

Random.seed!(0)

# A Matern 5/2 prior with lengthscale 0.2.
ℓ = 0.2
k = HalfIntegerMaternKernel(2, [ℓ])
f = GP(k)

# --- Three kinds of observations -----------------------------------------

# Point observations
X_pts = [0.05, 0.30, 0.55, 0.80, 0.95]
y_pts = f_true.(X_pts) .+ 0.02 .* randn(length(X_pts))
δ_pts = EvaluationFunctional(X_pts)
obs_pts = LinearObservation(δ_pts, y_pts; noise = 0.02^2)

# Derivative observations  ( ℒ = δ ∘ ∂x )
X_der = [0.15, 0.45]
y_der = f_deriv.(X_der) .+ 0.05 .* randn(length(X_der))
∂x = PartialDerivative((1,))
ℒ_der = EvaluationFunctional(X_der) ∘ ∂x
obs_der = LinearObservation(ℒ_der, y_der; noise = 0.05^2)

# Integral observations over two sub-intervals
intervals = [Interval(0.0, 0.5), Interval(0.5, 1.0)]
y_int = [0.0, 0.0]   # ∫₀^½ sin(2πx) dx = 1/π  ≈ 0.318
                     # ∫_½¹ sin(2πx) dx = -1/π ≈ -0.318
y_int .= [1/π, -1/π] .+ 0.01 .* randn(2)
ℒ_int = IntegralFunctional(intervals)
obs_int = LinearObservation(ℒ_int, y_int; noise = 0.01^2)
```

Condition the GP on all three observation types — order does not matter,
each call returns a posterior GP you can keep conditioning on:

```julia
f_post = condition_on_observation(f, obs_pts)
f_post = condition_on_observation(f_post, obs_der)
f_post = condition_on_observation(f_post, obs_int)
```

Predict and evaluate:

```julia
X_test = range(0, 1; length = 101)
μ, σ² = mean_and_var(f_post(collect(X_test)))
```

That's it — `μ` is the posterior mean, `σ²` is the posterior variance, and
`f_post` behaves like any other `AbstractGPs` posterior: you can sample from
it, push it through another functional, condition it on more data, and so on.

## What just happened?

The library walked through the composition you wrote and assembled the joint
covariance between every pair of observations:

- `δ_pts(δ_pts(k))` — the standard kernel matrix.
- `(δ ∘ ∂x)(δ_pts(k))` — derivative-vs-point cross-covariance, built from
  the analytic derivative of the Matérn kernel.
- `ℒ_int(δ_pts(k))` — integral-vs-point cross-covariance, built from the
  closed-form radial antiderivative of the Matérn kernel.
- `ℒ_int(ℒ_int(k))` — integral-vs-integral covariance, double-integrated
  via the same antiderivative chain.

For stationary kernels on regular grids, those matrices come back as lazy
Toeplitz objects; for compactly supported kernels they come back sparse.
You don't see this on the surface, but it's why mixing observation types is
practical at non-trivial sizes.

## Next steps

- For models where you need the **joint Gaussian over several named
  functionals** of the same GP — e.g. inside a Turing or DynamicPPL model
  for hyperparameter sampling — see [Joint Functional Gaussians](api/functional_gaussian.md).
- For the math-flavoured shorthand (`δ`, `∂`, `∫`), see [Notation](api/notation.md).
- For the full set of functionals, operators, and composition rules, see
  the [Functionals API reference](api/functionals.md).
- For the trait system that drives the fast paths under the hood, see
  [Specializations](api/specializations.md).
