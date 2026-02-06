# 2D Matérn GP Regression with Mixed Grid and Scattered Observations
#
# Demonstrates the tensor product kernel machinery with:
# - Grid observations on a FactorizedGrid (Kronecker-structured covariance)
# - Scattered observations at arbitrary points (Khatri-Rao / Hadamard cross-covariance)
# - Predictions on a fine grid
#
# The key structural result: when a tensor product kernel is used, the different
# observation types automatically produce efficient lazy matrix representations
# (Kronecker, Khatri-Rao, Hadamard) instead of dense matrices.

using FunctionalGPs
using AbstractGPs
using LinearAlgebra
using Random
using CairoMakie

# ============================================================================
# Ground truth
# ============================================================================

f_true(x, y) = sin(2π * x) * cos(π * y)

# ============================================================================
# GP prior: tensor product of 1D Matérn 5/2 kernels
# ============================================================================

k_x = HalfIntegerMaternKernel(2, [0.3])
k_y = HalfIntegerMaternKernel(2, [0.5])
k = k_x ⊗ k_y

f = GP(k)

# ============================================================================
# Observation 1: Grid observations (FactorizedGrid)
# ============================================================================

# Coarse grid — the covariance block for these observations has Kronecker
# structure, with each factor being a lazy SymmetricToeplitz matrix.
g_x = 0.1:0.2:0.9  # 5 points
g_y = 0.1:0.25:0.9  # 4 points (different spacing to show generality)
grid = FactorizedGrid(g_x, g_y)
n_grid_obs = prod(size(grid))

y_grid = [f_true(x, y) for y in g_y for x in g_x]

obs_grid = LinearObservation(
    EvaluationFunctional(grid),
    y_grid;
    noise = 1.0e-4,
)

println("Grid observations: $(length(g_x))×$(length(g_y)) = $(n_grid_obs) points")

# ============================================================================
# Observation 2: Scattered observations (vector-of-vectors)
# ============================================================================

# Arbitrary points — the cross-covariance between grid and scattered observations
# is a KhatriRaoMatrix; the scattered-scattered block is a Hadamard product
# (BroadcastArray) of per-dimension lazy kernel matrices.
Random.seed!(42)
n_scatter = 15
X_scatter = [[rand(), rand()] for _ in 1:n_scatter]
y_scatter = [f_true(x[1], x[2]) for x in X_scatter]

obs_scatter = LinearObservation(
    EvaluationFunctional(X_scatter),
    y_scatter;
    noise = 1.0e-4,
)

println("Scattered observations: $(n_scatter) points")

# ============================================================================
# Condition GP on both observation sets
# ============================================================================

println("\nConditioning on grid observations...")
f_post = condition_on_observation(f, obs_grid)
println("Conditioning on scattered observations...")
f_post = condition_on_observation(f_post, obs_scatter)
println("Done!")

# ============================================================================
# Predict on a fine grid
# ============================================================================

n_pred = 50
xs = range(0, 1; length = n_pred)
ys = range(0, 1; length = n_pred)
X_test = [[x, y] for y in ys for x in xs]

println("\nComputing posterior mean and variance at $(n_pred)×$(n_pred) = $(n_pred^2) points...")
μ, σ² = mean_and_var(f_post(X_test))
μ_grid = reshape(μ, n_pred, n_pred)
σ_grid = reshape(sqrt.(σ²), n_pred, n_pred)
f_true_grid = [f_true(x, y) for x in xs, y in ys]
err_grid = abs.(μ_grid .- f_true_grid)

println("Max absolute error: $(round(maximum(err_grid); sigdigits = 3))")
println("Max posterior std:  $(round(maximum(σ_grid); sigdigits = 3))")

# ============================================================================
# Visualization
# ============================================================================

fig = Figure(; size = (1400, 800))

# --- Row 1: True function, Posterior mean, Posterior std ---

ax1 = Axis(fig[1, 1]; title = "True function", xlabel = "x", ylabel = "y", aspect = 1)
hm1 = heatmap!(ax1, xs, ys, f_true_grid; colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3]; title = "Posterior mean", xlabel = "x", ylabel = "y", aspect = 1)
hm2 = heatmap!(ax2, xs, ys, μ_grid; colormap = :viridis)
# Grid observations (squares)
scatter!(
    ax2,
    [x for x in g_x for _ in g_y],
    [y for _ in g_x for y in g_y];
    color = :white, markersize = 10, marker = :rect, strokewidth = 1, strokecolor = :black,
)
# Scattered observations (circles)
scatter!(
    ax2,
    [x[1] for x in X_scatter],
    [x[2] for x in X_scatter];
    color = :red, markersize = 8, strokewidth = 1, strokecolor = :black,
)
Colorbar(fig[1, 4], hm2)

ax3 = Axis(
    fig[1, 5]; title = "Posterior std dev", xlabel = "x", ylabel = "y", aspect = 1,
)
hm3 = heatmap!(ax3, xs, ys, σ_grid; colormap = :plasma)
scatter!(
    ax3,
    [x for x in g_x for _ in g_y],
    [y for _ in g_x for y in g_y];
    color = :white, markersize = 10, marker = :rect, strokewidth = 1, strokecolor = :black,
)
scatter!(
    ax3,
    [x[1] for x in X_scatter],
    [x[2] for x in X_scatter];
    color = :red, markersize = 8, strokewidth = 1, strokecolor = :black,
)
Colorbar(fig[1, 6], hm3)

# --- Row 2: Absolute error ---

ax4 = Axis(
    fig[2, 1]; title = "Absolute error", xlabel = "x", ylabel = "y", aspect = 1,
)
hm4 = heatmap!(ax4, xs, ys, err_grid; colormap = :inferno)
scatter!(
    ax4,
    [x for x in g_x for _ in g_y],
    [y for _ in g_x for y in g_y];
    color = :white, markersize = 10, marker = :rect, strokewidth = 1, strokecolor = :black,
)
scatter!(
    ax4,
    [x[1] for x in X_scatter],
    [x[2] for x in X_scatter];
    color = :cyan, markersize = 8, strokewidth = 1, strokecolor = :black,
)
Colorbar(fig[2, 2], hm4)

# Legend
Legend(
    fig[2, 3:6],
    [
        MarkerElement(;
            color = :white, marker = :rect, markersize = 12,
            strokewidth = 1, strokecolor = :black,
        ),
        MarkerElement(;
            color = :red, marker = :circle, markersize = 10,
            strokewidth = 1, strokecolor = :black,
        ),
    ],
    ["Grid observations (FactorizedGrid)", "Scattered observations"];
    orientation = :horizontal, framevisible = false,
)

save("2d_matern_mixed_grid_scatter.png", fig)
println("\nSaved figure to 2d_matern_mixed_grid_scatter.png")

fig
