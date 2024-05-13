### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ ae190a88-e6bf-11ee-07d5-37f6f83b6421
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using GaussPDE
end

# ╔═╡ 8230b58f-088e-49b3-8a81-b355e1e569a3
begin
	# Activate our plotting environment
	using WGLMakie
	WGLMakie.activate!()
end

# ╔═╡ 1c9fcd58-5c09-4fa9-90b8-bdc95225da7b
begin
	using Random
	rng = MersenneTwister(1234)

	fps = 30
	ts = 0:1/fps:2
	frame_idcs = 1:2fps
	
	X_eval = FactorizedGrid(ts, 0:0.05:1)
end

# ╔═╡ 63056d42-9936-4c66-a6c2-317c99202a7c
md"""
# Problem Definition
In this notebook, we want to simulate a **1D Heat Equation** IBVP. It describes how heat diffuses through a given region.

The 1D heat equation is
```math
\frac{\partial u(t, x)}{\partial t} = \alpha \cdot \frac{\partial^2}{\partial x^2} u(t, x),
```
where the thermal diffusivity $\alpha$ is given by the thermal conductivity $\kappa$, specific heat capacity $c$ and density $\rho$ as follows:
```math
\alpha = \frac{\kappa}{\rho \cdot c}.
```
"""

# ╔═╡ 62fd1a2f-fd1d-48a0-a426-311b088e4338
md"""
For the initial condition, we prescribe the shape of the heat distribution at $t = 0$ through a truncated sine series:
```math
u(0, x) = \sum_{i=1}^4 \beta_i \sin(\pi \cdot i \cdot \frac{x+1}{2}).
```
"""

# ╔═╡ 82e32348-4592-43e9-92a7-318b1dac939a
md"""
At the boundary, we prescribe Dirichlet boundary conditions:
```math
u(t, x) = 0 \quad (x \in \partial \mathbb{X})
```
"""

# ╔═╡ 105d7ce3-1d8f-4828-a5b2-860d14d1ac9d
md"""
We do all of this on a rectangular domain.

GaussPDE has a pre-implemented utility function for this problem setup. It includes an analytic solution to compare our solver against.
"""

# ╔═╡ 826a9b82-38ae-4344-baf6-2ba2c808c239
problem = Heat1DIBVPTruncatedSineICDirichletBC(
    BoxDomain((0.0, 2.0), (0.0, 1.0)),
    c=1.0,
    ρ=1.0,
    κ=0.05,
    ic_coeffs=[1, 1//2, 1//4, 1//8],
)

# ╔═╡ 3b9c23e5-5be4-41f4-836f-e4470e9a8fd7
soln = solution(problem)

# ╔═╡ 37291a78-23f3-47e2-9e1e-964e796c4313
lines(0:0.01:1, soln(convert(Array, FactorizedGrid(0:0.1:2, 0:0.01:1)))[1, :])

# ╔═╡ 72858e60-53c2-43ff-a90d-6e96d36c3a1f
md"""
# Choosing a prior
First, we need to choose a prior for our problem.

Typically, we model multiple inputs with tensor product kernels.

Then, for each kernel, we essentially need to make two choices. i) How smooth should it be, and ii) which lengthscale should it have?

i) is typically pretty simple: You use the minimum smoothness required by the PDE.

ii) is often more involved - it depends on both the initial condition and the PDE dynamics. Rapidly evolving solutions need smaller lengthscales, for example. You may want to use hyperparameter optimization tools for this.
"""

# ╔═╡ 9c26f8a2-ca79-4e4b-b0f7-24799dacb1a5
begin
	smoothness_t = 1 # We need to differentiate once across time
	smoothness_x = 2 # and twice across space
	# Lengthscale 1.0 is the starting point of uncreative people like myself :)
	ℓ_t = 1.0
	ℓ_x = 1.0
end

# ╔═╡ 79c7897b-2e93-44c5-95e7-3ab9fdc9eb54
begin
	k_t = WendlandKernel(1, smoothness_t, ℓ_t)
	k_x = WendlandKernel(1, smoothness_x, ℓ_x)
	k = k_t ⊗ k_x
end

# ╔═╡ 1fe78b4b-a580-4742-a61d-27956a7420be
begin
	using AbstractGPs

	f = GP(k)
end

# ╔═╡ 29c40c0a-68c0-43a9-a567-933dd82cf6a4
md"""
Let's now sample 20 points from the initial condition.
"""

# ╔═╡ d6e12f3e-66a8-4b50-a15e-7a42b89d29e5
begin
	ic_obs = sample_ic(problem, 20)
	X_ic = ic_obs.ℒ.X
	Y_ic = ic_obs.y
	size(X_ic), size(Y_ic)
end

# ╔═╡ bb20914a-32a5-4015-985c-66e0332b1c72
md"""
`X_ic` contains the locations, and `Y_ic` contains the corresponding values.

`X_ic` has size (1, 20) because we have 1 time point (t = 0) and 20 points in space. It is a variable of type `FactorizedGrid`, which enables efficient routines for `TensorProductKernel`s.
"""

# ╔═╡ 31d840f7-2787-4b9a-87b6-b3936e4064bc
md"""
What about `ic_obs.ℒ`? `ic_obs.ℒ` is the linear operator that corresponds to evaluating the GP at the points given by `X_ic`.
"""

# ╔═╡ d879af42-bbfe-49a8-b99f-93f88ac6757e
ic_obs.ℒ

# ╔═╡ 127b529b-bef4-4925-88d6-7e6487138ba9
md"""
This is all encapsulated in an object of type `LinearObservation`.
"""

# ╔═╡ 44c2b339-45da-45d5-999e-b4fa30a9e6f0
ic_obs

# ╔═╡ 83153242-67fe-4dcd-98eb-365ad7ddc6be
md"""
We can condition GPs on these types of observations! Let's try it:
"""

# ╔═╡ 3a183fce-b4af-47e9-916d-bc1fd84f963c
f_ic = condition_on_observation(f, ic_obs)

# ╔═╡ 5ad6fdbf-5e94-42cf-912e-a751017176e2
md"""
# Interlude: Plotting
Let's write some plotting code. This will let us drag a slider to move through time.
"""

# ╔═╡ 006882fb-53e6-4171-b6aa-2e8773d24c61
function plot_time_slice(box, f_cond, X_eval, frame_idx)
    sols = solution(problem)(convert(Array, X_eval))
    f_cond_eval = f_cond(X_eval)
    means, vars = mean_and_var(f_cond_eval)
    means = reshape(means, size(X_eval))
    vars = reshape(vars, size(X_eval))
    stds = sqrt.(vars)
    y_lim_upper = maximum(means .+ 1.96 * stds)
    y_lim_lower = minimum(means .- 1.96 * stds) - 0.1

    f_ax = Axis(box, limits=(0, 1.0, y_lim_lower, y_lim_upper))

    samples = rand(rng, f_cond_eval, 3)
    samples = reshape(samples, size(X_eval)..., 3)

    # f_ic_mean, f_ic_var = mean_and_var(f_ic(X_ic))
    mean_t = @lift(means[$frame_idx, :])
    std_t = @lift(stds[$frame_idx, :])
    lines!(f_ax, X_eval[2], mean_t, color=:blue)
    conf = @lift(1.96 * $std_t)
    upper = @lift($mean_t .+ $conf)
    lower = @lift($mean_t .- $conf)
    band!(f_ax, X_eval[2], lower, upper, color=:blue, alpha=0.3)
    for i in axes(samples, 3)
        cur_sample_i = @lift(samples[$frame_idx, :, i])
        lines!(f_ax, X_eval[2], cur_sample_i, color=:gray, alpha=0.3)
    end
    sol_vals = @lift(sols[$frame_idx, :])
    lines!(f_ax, X_eval[2], sol_vals, color=:gold, linestyle=:dash)
    return f_ax
end

# ╔═╡ 5071d138-ac83-48b3-b6ef-b8cbee811770
function plot_interactive(fig, f_cond)
	time_idx = Observable(1)
	plot_time_slice(fig[1, 1], f_cond, X_eval, time_idx)

	time_slider = Makie.Slider(fig[2, 1], range=frame_idcs, startvalue=1)
	on(time_slider.value) do val
	    time_idx[] = val
	end
end

# ╔═╡ 9434bd98-8090-4084-9690-1a2e5d294c53
md"""
# Moving on...
Here's the visualization of `f_ic`:
"""

# ╔═╡ a228e690-96bd-4a7e-b14c-8bafb025b236
begin
	fig = Figure()
	plot_interactive(fig, f_ic)
	fig
end

# ╔═╡ 90f07b7f-ad67-4c91-b0e2-38454e8cd753
md"""
Nice, that looks reasonable!

Now let's further constrain the function to be zero at the boundaries.
"""

# ╔═╡ 07a4209f-8951-4ecd-9f9b-7d70adea093b
begin
	bc_obs = sample_bc(problem, 40)
	f_ic_bc = condition_on_observation(f_ic, bc_obs)
end

# ╔═╡ ec1cdcca-ee8f-4b34-8f8c-64b7975541ef
begin
	fig_ic_bc = Figure()
	plot_interactive(fig_ic_bc, f_ic_bc)
	fig_ic_bc
end

# ╔═╡ 0a9a014a-289e-4796-aeab-03f2c5786af0
begin
	# Ignore this - it's a hotfix
	import KernelFunctions: ScaledKernel
	
	function ScaledKernel(kernel::Tk, σ²::Tσ²=1.0) where {Tk<:Kernel,Tσ²<:Real}
	    return ScaledKernel{Tk,Tσ²}(kernel, [σ²])
	end
end

# ╔═╡ 34f671cb-0024-46d3-b4ca-b9b69dc201c2
md"""
Finally, it's time for the part  you've been waiting for: Conditioning on PDE information.

We are going to use collocation observations. This means that we are going to fulfill $\mathcal{D}[u](x_i) = f(x_i)$ exactly for all $x_i$ ($i \in \{1, \dots, n\}$). The more observations we use (i.e., the larger n gets), the better our model of the solution will be. This will also shrink the uncertainty.

Without further ado, let's try it out:
"""

# ╔═╡ b46d7a98-e858-4586-961a-8badaf29a514
begin
	N_t = 20
	N_x = 10
	𝒟 = lindiffops(problem)[1] # Heat operator
	X_pde = FactorizedGrid(range(0, 2, N_t), range(0, 1, N_x)) # Evaluation grid
	ℒ = EvaluationFunctional(X_pde) ∘ 𝒟 # Concatenate 𝒟 with evaluation operator
	Y_pde = zeros(length(X_pde)) # Sparse vector of all zeros

	f_all = condition_on_observation(f_ic_bc, ℒ, Y_pde, noise=1e-8)
end

# ╔═╡ 9d3df66b-fc60-40b4-ad5e-24d9a3786ee1
begin
	using SparseArrays

	Δt = 0.1
	Δx = 0.2
	# PDE
	intervals_t = intervals_from_endpoints(0:Δt:2)
	intervals_x = intervals_from_endpoints(0:Δx:1)
	box_domains = intervals_t ⊗ intervals_x
	∫ = VectorizedLebesgueIntegral(box_domains)
	
	ℒ_FVM = ∫ ∘ 𝒟
	Y_FVM = spzeros(length(box_domains))
	output_shape(ℒ_FVM)
end

# ╔═╡ 2d90f43b-b399-42f3-a6a0-d591e553cc9a
begin
	fig_all = Figure()
	plot_interactive(fig_all, f_all)
	fig_all
end

# ╔═╡ fda4c256-6396-4566-8e04-017b2ff0d9e7
md"""
It's better than before, but still not satisfactory. Let's use more collocation observations!
"""

# ╔═╡ a3f9f454-f8a8-4328-b535-948e68125c31
begin
	N_t_moar = 50
	N_x_moar = 60
	X_pde_moar = FactorizedGrid(range(0, 2, N_t_moar), range(0, 1, N_x_moar))
	ℒ_moar = EvaluationFunctional(X_pde_moar) ∘ 𝒟
	Y_pde_moar = zeros(length(X_pde_moar))

	f_moar = condition_on_observation(f_ic_bc, ℒ_moar, Y_pde_moar, noise=1e-8)
end

# ╔═╡ 102ccd58-cb48-4fe3-82ee-a3a24c63847b
begin
	fig_moar = Figure()
	plot_interactive(fig_moar, f_moar)
	fig_moar
end

# ╔═╡ 42468cf3-687a-4741-a2ce-9ff72009f7cb
md"""
# Finite Volume Method
That's cool! Seems a bit wasteful though to use that many observations for such a simple problem. Can we do better?

Let's try volumetric observations. For further information on this, ask me (Tim) :)
"""

# ╔═╡ 84c7c360-50f2-42c4-b87f-2109d3612832
f_FVM = condition_on_observation(f_ic_bc, ℒ_FVM, Y_FVM, noise=1e-8)

# ╔═╡ f381d8c1-a466-45c0-ae09-18112521b2aa
begin
	fig_FVM = Figure()
	plot_interactive(fig_FVM, f_FVM)
	fig_FVM
end

# ╔═╡ dbd9e196-85b0-4508-8c28-b43ad358520b
md"""
As you may have noticed, this uses half as many observations as the first collocation attempt. And yet, it produces a much more sensible posterior! This is the power of the Finite Volume Method. 🎉
"""

# ╔═╡ Cell order:
# ╠═ae190a88-e6bf-11ee-07d5-37f6f83b6421
# ╟─63056d42-9936-4c66-a6c2-317c99202a7c
# ╟─62fd1a2f-fd1d-48a0-a426-311b088e4338
# ╟─82e32348-4592-43e9-92a7-318b1dac939a
# ╟─105d7ce3-1d8f-4828-a5b2-860d14d1ac9d
# ╟─826a9b82-38ae-4344-baf6-2ba2c808c239
# ╠═3b9c23e5-5be4-41f4-836f-e4470e9a8fd7
# ╠═8230b58f-088e-49b3-8a81-b355e1e569a3
# ╠═37291a78-23f3-47e2-9e1e-964e796c4313
# ╟─72858e60-53c2-43ff-a90d-6e96d36c3a1f
# ╠═9c26f8a2-ca79-4e4b-b0f7-24799dacb1a5
# ╠═79c7897b-2e93-44c5-95e7-3ab9fdc9eb54
# ╠═1fe78b4b-a580-4742-a61d-27956a7420be
# ╟─29c40c0a-68c0-43a9-a567-933dd82cf6a4
# ╠═d6e12f3e-66a8-4b50-a15e-7a42b89d29e5
# ╟─bb20914a-32a5-4015-985c-66e0332b1c72
# ╟─31d840f7-2787-4b9a-87b6-b3936e4064bc
# ╠═d879af42-bbfe-49a8-b99f-93f88ac6757e
# ╟─127b529b-bef4-4925-88d6-7e6487138ba9
# ╠═44c2b339-45da-45d5-999e-b4fa30a9e6f0
# ╟─83153242-67fe-4dcd-98eb-365ad7ddc6be
# ╠═3a183fce-b4af-47e9-916d-bc1fd84f963c
# ╟─5ad6fdbf-5e94-42cf-912e-a751017176e2
# ╠═1c9fcd58-5c09-4fa9-90b8-bdc95225da7b
# ╠═006882fb-53e6-4171-b6aa-2e8773d24c61
# ╠═5071d138-ac83-48b3-b6ef-b8cbee811770
# ╟─9434bd98-8090-4084-9690-1a2e5d294c53
# ╟─a228e690-96bd-4a7e-b14c-8bafb025b236
# ╟─90f07b7f-ad67-4c91-b0e2-38454e8cd753
# ╠═07a4209f-8951-4ecd-9f9b-7d70adea093b
# ╟─ec1cdcca-ee8f-4b34-8f8c-64b7975541ef
# ╟─0a9a014a-289e-4796-aeab-03f2c5786af0
# ╟─34f671cb-0024-46d3-b4ca-b9b69dc201c2
# ╠═b46d7a98-e858-4586-961a-8badaf29a514
# ╟─2d90f43b-b399-42f3-a6a0-d591e553cc9a
# ╟─fda4c256-6396-4566-8e04-017b2ff0d9e7
# ╠═a3f9f454-f8a8-4328-b535-948e68125c31
# ╟─102ccd58-cb48-4fe3-82ee-a3a24c63847b
# ╟─42468cf3-687a-4741-a2ce-9ff72009f7cb
# ╠═9d3df66b-fc60-40b4-ad5e-24d9a3786ee1
# ╠═84c7c360-50f2-42c4-b87f-2109d3612832
# ╟─f381d8c1-a466-45c0-ae09-18112521b2aa
# ╟─dbd9e196-85b0-4508-8c28-b43ad358520b
