# Joint Functional Gaussians

`FunctionalGaussian` represents the joint Gaussian induced by applying a named
collection of linear functionals to a Gaussian process. It owns the joint
mean, the joint covariance (with per-block storage that preserves structured
`CovarianceMatrix` subtypes), and the algebra of marginalising, conditioning,
and likelihood evaluation.

It exists because writing one independent `~ MvNormal(...)` per block in a
probabilistic programming model would drop the cross-covariances induced by
the shared underlying GP. `FunctionalGaussian` collects all functionals into
a single joint object so those cross-covariances are preserved.

## Construction

```julia
using FunctionalGPs, AbstractGPs

f = GP(WendlandKernel(1, 3, 8 // 10))

X_obs  = collect(0.0:0.1:1.0)
X_pred = collect(0.0:0.05:1.0)

fg = FunctionalGaussian(f;
    y  = EvaluationFunctional(X_obs),
    dy = EvaluationFunctional(X_pred) ∘ PartialDerivative((1,)),
    q  = VectorizedLebesgueIntegral([Interval(0.0, 1.0)]),
)
```

The same construction reads as math via the `Notation` submodule (see the
**Notation** page): `δ(X_obs)`, `δ(X_pred) ∘ ∂(1)`, `∫([Interval(0, 1)])`.

```@docs
FunctionalGaussian
```

## Block accessors

Each named block has a row/column range inside the joint mean and covariance.
Per-block accessors return the underlying objects unchanged, so structured
`CovarianceMatrix` subtypes (Toeplitz, Khatri-Rao, sparse, ...) keep their
fast `getindex`/matvec paths.

```julia
keys(fg)              # (:y, :dy, :q)
length(fg)            # total dimension
block_range(fg, :dy)  # row/col range in the joint Σ
mean(fg)              # joint mean Vector
mean(fg, :dy)         # mean of one block (view)
cov(fg)               # joint Σ (BlockMatrix; per-block types retained)
cov(fg, :y)           # self-covariance of :y, e.g. StationaryKernelMatrix
cov(fg, :y, :dy)      # cross-covariance — same structure preservation
```

```@docs
block_range
```

## Property access

`fg.<block_name>` returns a [`LazyMvNormal`](@ref) over that block: an
`AbstractMvNormal` that does *not* eagerly factorise its covariance, so the
structured per-block matrix flows through to `cov(fg.y)` unchanged.

```julia
fg.y                 # LazyMvNormal{..., StationaryKernelMatrix}
mean(fg.y)
cov(fg.y)            # same object as cov(fg, :y) — no factorisation
var(fg.y)
```

Block-name access takes priority over the underlying struct fields. To access
an internal field that is shadowed by a block name (e.g. a block named `:μ`),
use `getfield(fg, :μ)`.

`propertynames(fg)` returns the block names by default; pass `true` to also
see the struct fields.

## LazyMvNormal

```@docs
LazyMvNormal
```

`LazyMvNormal` is structure-preserving for `mean` / `cov` / `var` / `length`.
For repeated `logpdf` / `rand` calls (e.g. in a tight inference loop), build
the eager `MvNormal` once via `MvNormal(d)` to amortise the Cholesky.

## Marginalisation

```julia
marginal(fg, :dy)            # equivalent to fg.dy
marginal(fg, (:y, :q))       # multi-block, order respected
```

Multi-block marginals materialise a dense joint covariance (the Cholesky for
joint sampling needs it). Single-block marginals keep the structured per-block
matrix.

```@docs
marginal
```

## Conditioning — `posterior`

`FunctionalGPs` extends [`AbstractGPs.posterior`](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/api/#AbstractGPs.posterior).
For a `FunctionalGaussian`, posterior conditions the joint Gaussian on
observed values for a subset of named blocks and returns a NamedTuple of
[`LazyMvNormal`](@ref) over the remaining blocks.

```julia
post = posterior(fg, (; y = y_obs); noise = (; y = σ²))
post.dy        # LazyMvNormal over derivative locations
post.q         # LazyMvNormal over the integral
```

`condition` uses block-aware matmul for the cross-block `Σ_lo * (C \\ residual)`
products, so per-block fast matvec paths (Toeplitz, Khatri-Rao, …) are
exploited automatically when the kernel admits them.

`noise` accepts the same forms as [`LinearObservation`](@ref): scalar variance,
vector of variances, or full covariance matrix. Element types propagate, so
`ForwardDiff.Dual` and `BigFloat` hyperparameters trace through cleanly.

```@docs
AbstractGPs.posterior(::FunctionalGaussian, ::NamedTuple)
```

## Marginal log-likelihood

```julia
ℓ = loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))
```

This is the single hook for hyperparameter inference. Inside a Turing model:

```julia
@model function gp_inference(y_obs, X_obs)
    ell ~ LogNormal(log(0.2), 0.5)
    σ² ~ LogNormal(log(0.01), 1.0)
    f = GP(with_lengthscale(SqExponentialKernel(), ell))
    fg = FunctionalGaussian(f; y = EvaluationFunctional(X_obs))
    Turing.@addlogprob! loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))
    return posterior(fg, (; y = y_obs); noise = (; y = σ²))
end
```

A complete runnable example with NUTS hyperparameter sampling lives at
`examples/turing_hyperparams/turing_inference.jl`.

```@docs
FunctionalGPs.loglikelihood(::FunctionalGaussian, ::NamedTuple)
```

## Joint as a flat MvNormal

```@docs
as_mvn
```

`as_mvn(fg)` materialises the full joint covariance and runs Cholesky eagerly,
returning a regular `Distributions.MvNormal`. Reach for it only when you need
a strictly Distributions-compatible flat object — for everything else, prefer
the `LazyMvNormal` paths above.
