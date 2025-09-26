using GaussPDE
using AbstractGPs
using WGLMakie
using Random
using KernelFunctions
import KernelFunctions: ScaledKernel
using SparseArrays

function ScaledKernel(kernel::Tk, σ²::Tσ² = 1.0) where {Tk <: Kernel, Tσ² <: Real}
    return ScaledKernel{Tk, Tσ²}(kernel, [σ²])
end

WGLMakie.activate!()
fig = Figure()
rng = MersenneTwister(1234)

# PRIOR
l_t, l_x = 1.0, 1.0
k_t = WendlandKernel(1, 1, l_t)
k_x = WendlandKernel(1, 2, l_x)
k = k_t ⊗ k_x
f = GP(k)

problem = Heat1DIBVPTruncatedSineICDirichletBC(
    BoxDomain((0.0, 2.0), (0.0, 1.0)),
    c = 1.0,
    ρ = 1.0,
    κ = 0.05,
    ic_coeffs = [1, 1 // 2, 1 // 4, 1 // 8],
)

# INITIAL CONDITION
ic_obs = sample_ic(problem, 20)
X_ic = ic_obs.ℒ.X
Y_ic = ic_obs.y
ic_ax = Axis(fig[1, 1])
lines!(ic_ax, X_ic[2], Y_ic, color = :blue)
f_ic = condition_on_observation(f, ic_obs)


fps = 30
ts = 0:(1 / fps):2
frame_idcs = 1:2fps

X_eval = FactorizedGrid(ts, 0:0.05:1)

function plot_time_slice(box, f_cond, X_eval, frame_idx)
    sols = solution(problem)(convert(Array, X_eval))
    f_cond_eval = f_cond(X_eval)
    means, vars = mean_and_var(f_cond_eval)
    means = reshape(means, size(X_eval))
    vars = reshape(vars, size(X_eval))
    stds = sqrt.(vars)
    y_lim_upper = maximum(means .+ 1.96 * stds)
    y_lim_lower = minimum(means .- 1.96 * stds) - 0.1

    f_ax = Axis(box, limits = (0, 1.0, y_lim_lower, y_lim_upper))

    samples = rand(rng, f_cond_eval, 3)
    samples = reshape(samples, size(X_eval)..., 3)

    # f_ic_mean, f_ic_var = mean_and_var(f_ic(X_ic))
    mean_t = @lift(means[$frame_idx, :])
    std_t = @lift(stds[$frame_idx, :])
    lines!(f_ax, X_eval[2], mean_t, color = :blue)
    conf = @lift(1.96 * $std_t)
    upper = @lift($mean_t .+ $conf)
    lower = @lift($mean_t .- $conf)
    band!(f_ax, X_eval[2], lower, upper, color = :blue, alpha = 0.3)
    for i in axes(samples, 3)
        cur_sample_i = @lift(samples[$frame_idx, :, i])
        lines!(f_ax, X_eval[2], cur_sample_i, color = :gray, alpha = 0.3)
    end
    sol_vals = @lift(sols[$frame_idx, :])
    lines!(f_ax, X_eval[2], sol_vals, color = :gold, linestyle = :dash)
    return f_ax
end

time_idx = Observable(1)
f_ic_ax = plot_time_slice(fig[1, 2], f_ic, X_eval, time_idx)

# BOUNDARY CONDITION
bc_obs = sample_bc(problem, 40)
f_ic_bc = condition_on_observation(f_ic, bc_obs)

f_ic_bc_ax = plot_time_slice(fig[2, 1], f_ic_bc, X_eval, time_idx)

# PDE
𝒟 = lindiffops(problem)[1]
X_pde = FactorizedGrid(0:0.03:2, 0:0.01:1)
ℒ = EvaluationFunctional(X_pde) ∘ 𝒟
Y_pde = spzeros(length(X_pde))

f_all = condition_on_observation(f_ic_bc, ℒ, Y_pde, noise = 1.0e-8)

f_all_ax = plot_time_slice(fig[2, 2], f_all, X_eval, time_idx)

time_slider = Makie.Slider(fig[3, 1:2], range = frame_idcs, startvalue = 1)
on(time_slider.value) do val
    time_idx[] = val
end
