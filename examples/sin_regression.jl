using GaussPDE
using AbstractGPs
using Plots

k = WendlandKernel(1, 3, 8 // 10)
f = GP(k)

X = [3.14]
y = sin.(X)

f_1 = condition_on_observation(f, X, y, noise = 1.0e-8)

# X_cos = sort(rand(0:0.1:2π, 100))
X_cos = 0:0.1:2π
dx = PartialDerivative((1,))
ℒ = EvaluationFunctional(X_cos) ∘ dx

f_2 = condition_on_observation(f_1, ℒ, cos.(X_cos), noise = 1.0e-9)

X_test = 0:0.01:2π
plot(X_test, f_2)
plot!(X_test, rand(f_2(X_test), 10); color = "gray", alpha = 0.4, label = nothing)
# Plot cos points
scatter!(X_cos, cos.(X_cos); label = "cos points")
savefig("test.png")
