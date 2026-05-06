# FunctionalGaussian usage walk-through.
#
# A `FunctionalGaussian` represents the joint Gaussian induced by applying a
# named collection of linear functionals to a GP. Marginalising, conditioning,
# and likelihood evaluation operate on this single joint object, so
# cross-covariances between blocks are preserved (you do NOT get this if you
# write one `~ MvNormal(...)` per block in a Turing model).
#
# Run:
#     julia --project examples/functional_gaussian.jl

using FunctionalGPs
using AbstractGPs
using KernelFunctions
using LinearAlgebra
using Distributions

# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

# Use a stationary 1D kernel — its evaluation/derivative blocks come back as
# structured CovarianceMatrix subtypes (StationaryKernelMatrix etc.), which the
# rest of this example will leverage.
k = with_lengthscale(SqExponentialKernel(), 0.5)
f = GP(k)

X = collect(0.0:0.1:1.0)         # observation locations
Xd = collect(0.25:0.1:0.75)       # derivative-observation locations
Xq = [Interval(0.0, 1.0)]         # one integral over [0, 1]

L_y = EvaluationFunctional(X)
L_dy = EvaluationFunctional(Xd) ∘ PartialDerivative((1,))
L_q = VectorizedLebesgueIntegral(Xq)

fg = FunctionalGaussian(f; y = L_y, dy = L_dy, q = L_q)

# REPL display
println("=== show(fg) ===")
show(stdout, MIME("text/plain"), fg)
println()

# ---------------------------------------------------------------------------
# 2. Block bookkeeping
# ---------------------------------------------------------------------------

@show keys(fg)                    # (:y, :dy, :q)
@show length(fg)                  # total dimension
@show block_range(fg, :dy)        # row/col range inside the joint Σ
@show propertynames(fg)           # block names — clean tab-completion

# ---------------------------------------------------------------------------
# 3. Means and covariances
# ---------------------------------------------------------------------------

mean(fg)                          # joint mean (Vector)
cov(fg)                           # joint Σ (BlockMatrix; per-block storage retained)
mean(fg, :dy)                     # mean of one block (view)
cov(fg, :y)                       # self-covariance of :y — STRUCTURED type
cov(fg, :y, :dy)                  # cross-covariance between :y and :dy

println("\ntypeof(cov(fg, :y)) = ", typeof(cov(fg, :y)))   # StationaryKernelMatrix
println("typeof(cov(fg, :y, :dy)) = ", typeof(cov(fg, :y, :dy)))

# ---------------------------------------------------------------------------
# 4. Block marginals via `.` access — LazyMvNormal, structure-preserving
# ---------------------------------------------------------------------------

# fg.<name> returns a LazyMvNormal. It does NOT factorise the covariance, so
# the structured CovarianceMatrix flows through to `cov(fg.y)` unchanged.
y_dist = fg.y
@show typeof(y_dist)
@show typeof(cov(y_dist))                  # same StationaryKernelMatrix
@show cov(y_dist) === cov(fg, :y)          # true — same object

# Cheap, structure-preserving:  mean / cov / var / length / dim
mean(fg.y)
cov(fg.y)
var(fg.y)                                  # diag(Σ); fast for structured Σ

# Heavier ops (logpdf, rand) materialise + factorise on each call.
# For tight inference loops, build an MvNormal once and reuse:
mvn = MvNormal(fg.y)                       # eager Cholesky now, cached forever
logpdf(mvn, randn(length(fg.y)))           # subsequent calls reuse the factor

# ---------------------------------------------------------------------------
# 5. Marginalisation (single and multi-block)
# ---------------------------------------------------------------------------

# Equivalent to fg.dy
marg_dy = marginal(fg, :dy)
@assert mean(marg_dy) ≈ mean(L_dy(f))

# Multi-block marginal — order respected
marg_yq = marginal(fg, (:y, :q))
@show length(marg_yq)

# ---------------------------------------------------------------------------
# 6. Conditioning — observe a subset of blocks, get the rest as posteriors
# ---------------------------------------------------------------------------

y_obs = sin.(2π .* X) .+ 0.1 .* randn(length(X))
σ² = 0.01

post = condition(fg, (; y = y_obs); noise = (; y = σ²))
@show keys(post)                   # (:dy, :q) — y was observed, the rest are latent

# Each entry is a LazyMvNormal — same lazy/structured semantics
mean(post.dy)
cov(post.dy)
mean(post.q)

# Observe MULTIPLE blocks at once.  Useful when both function values and
# integral measurements are available simultaneously.
q_obs = [0.0]                      # ∫_0^1 sin(2πx) dx ≈ 0
post2 = condition(
    fg, (; y = y_obs, q = q_obs); noise = (; y = σ², q = 1.0e-8),
)
@show keys(post2)                  # (:dy,)

# Sanity-check against the GP-posterior route (condition_on_observation +
# applying L_dy to the resulting posterior GP):
f_post = condition_on_observation(f, L_y, y_obs; noise = σ²)
@assert isapprox(mean(post.dy), mean(L_dy(f_post)); atol = 1.0e-10)

# ---------------------------------------------------------------------------
# 7. Marginal log-likelihood — drop into Turing for hyperparameter inference
# ---------------------------------------------------------------------------

# Scalar variance noise:
ll = loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))

# Vector of variances (per-element heteroscedastic noise):
ll2 = loglikelihood(fg, (; y = y_obs); noise = (; y = fill(σ², length(y_obs))))

# Full noise covariance matrix:
ll3 = loglikelihood(
    fg, (; y = y_obs); noise = (; y = Matrix(σ² * I(length(y_obs)))),
)

@show ll ll2 ll3

# Turing sketch (commented out — needs Turing.jl):
#
# using Turing
#
# @model function derivative_regression(y_obs, X, Xd)
#     ell    ~ LogNormal(0, 1)
#     sigma2 ~ Exponential(1)
#     f      = GP(with_lengthscale(SqExponentialKernel(), ell))
#     fg     = FunctionalGaussian(f;
#         y  = EvaluationFunctional(X),
#         dy = EvaluationFunctional(Xd) ∘ PartialDerivative((1,)),
#     )
#     Turing.@addlogprob! loglikelihood(fg, (; y = y_obs); noise = (; y = sigma2))
#     return fg
# end
#
# chain = sample(derivative_regression(y_obs, X, Xd), NUTS(), 1000)

# ---------------------------------------------------------------------------
# 8. Joint as a flat MvNormal (eager — for code that needs Distributions.jl)
# ---------------------------------------------------------------------------

# `as_mvn(fg)` returns a regular `Distributions.MvNormal` over the full joint
# (length 18 here). It materialises the covariance and runs Cholesky eagerly,
# so it can hit pos-def issues for ill-conditioned joints (e.g. an SE kernel
# at fine spacing). Prefer `LazyMvNormal` paths above; reach for `as_mvn` only
# when you need a strictly Distributions-compatible flat object.
#
#     joint_mvn = as_mvn(fg)

println("\nDone.")
