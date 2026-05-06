# Hyperparameter inference for a FunctionalGaussian via Turing + NUTS.
#
# Setup:
#   julia --project=examples/turing_hyperparams -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#   julia --project=examples/turing_hyperparams examples/turing_hyperparams/turing_inference.jl
#
# What this validates:
#   - `loglikelihood(fg, ...)` traces through ForwardDiff inside @model.
#   - Hyperparameter posteriors recover sensible values from synthetic data.
#   - Returning `posterior(fg, ...)` from the model gives users named-block
#     posteriors (here: posterior over derivative locations) per sample.

using FunctionalGPs
using AbstractGPs
using KernelFunctions: with_lengthscale, SqExponentialKernel
using Distributions
using Turing
using Random
using Statistics

# ---------------------------------------------------------------------------
# Synthetic data: noisy observations of f(x) = sin(2π x) on [0, 1].
# ---------------------------------------------------------------------------

Random.seed!(42)
true_ell = 0.15
true_σ² = 0.05^2
N = 25
X_obs = sort!(rand(N))
y_obs = sin.(2π .* X_obs) .+ sqrt(true_σ²) .* randn(N)

# Derivative locations we want posteriors at (not observed).
X_pred_dy = collect(0.0:0.1:1.0)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@model function gp_inference(y_obs, X_obs, X_pred_dy)
    ell ~ LogNormal(log(0.2), 0.5)
    σ² ~ LogNormal(log(0.01), 1.0)

    f = GP(with_lengthscale(SqExponentialKernel(), ell))
    fg = FunctionalGaussian(
        f;
        y = EvaluationFunctional(X_obs),
        dy = EvaluationFunctional(X_pred_dy) ∘ PartialDerivative((1,)),
    )
    Turing.@addlogprob! loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))
    return posterior(fg, (; y = y_obs); noise = (; y = σ²))
end

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

model = gp_inference(y_obs, X_obs, X_pred_dy)

println("Sampling 4 chains × 500 NUTS iterations…")
chain = sample(model, NUTS(0.65), MCMCThreads(), 500, 4; progress = false)

println("\n--- Chain summary ---")
display(chain)

# Posterior summaries against ground truth
ell_post = vec(Array(chain[:ell]))
σ²_post = vec(Array(chain[:σ²]))
println("\nlengthscale  posterior mean = $(round(mean(ell_post); digits = 4))   true = $true_ell")
println("noise var    posterior mean = $(round(mean(σ²_post); digits = 6))   true = $true_σ²")

# ---------------------------------------------------------------------------
# Posterior over derivative at X_pred_dy via the returned conditional
# ---------------------------------------------------------------------------

posts = generated_quantities(model, chain)
dy_means = stack(getfield.(posts, :dy) .|> mean; dims = 2)
dy_mean = vec(mean(dy_means; dims = 2))
dy_std = vec(std(dy_means; dims = 2))

true_dy = 2π .* cos.(2π .* X_pred_dy)
println("\nDerivative posterior at X_pred_dy:")
println("  posterior mean: ", round.(dy_mean; digits = 3))
println("  posterior std:  ", round.(dy_std; digits = 3))
println("  truth (2π cos): ", round.(true_dy; digits = 3))
