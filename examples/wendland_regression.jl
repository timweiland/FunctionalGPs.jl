using FunctionalGPs
using AbstractGPs, Plots
import Random: seed!
using Permutations

seed!(3032456)
N = 20
noise = 0.001
plot_step = 0.01

x = sort(2π * rand(N))
y = sin.(x)

w = 0.01 * WendlandKernel(1, 6, 0.6)
f = GP(w)

# Finite projection of `f` at inputs `x`.
# Added Gaussian noise with variance 0.001.
fx = f(x, noise)

println(cov(fx))

# Exact posterior given `y`. This is another GP.
p_fx = posterior(fx, y)

x_test = sort(2π * rand(5))

K_test = kernelmatrix(w, x_test)
K_train = kernelmatrix(w, x) + noise * diagm(ones(length(x)))
crosscov = kernelmatrix(w, x_test, x)
post_cov_manual = K_test - crosscov * (cholesky(K_train) \ crosscov')
cov(p_fx(x_test))
# println(cov(p_fx(x)))

#sample
μ, Σ = mean_and_cov(p_fx(0:plot_step:2π, 0.0001))
C = cholesky(Σ)
samples = μ .+ (sparse(C.L) * randn(length(μ), 20))[invperm(C.p), :]
# plot(0:plot_step:2π, samples[:, 1]; label="Samples")

# # Plot posterior.
scatter(x, y; label = "Data")
plot!(0:plot_step:2π, samples; alpha = 0.2, color = "gray", label = "")
plot!(0:plot_step:2π, p_fx; label = "Posterior")
