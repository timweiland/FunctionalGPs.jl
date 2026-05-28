export Identity

"""
    Identity()

The identity linear function operator. Acts as a pass-through on kernels,
mean functions, and process-vector crosscovariances. Serves two roles in the
operator algebra:

- The unit of `∘`: `δ ∘ Identity()` behaves like `δ`.
- A zeroth-order term in sums: `Identity() + ∂x` acts as `k + ∂k/∂x` on a
  kernel, which is the natural shape for PDE residuals with a constant term.

# Example
```julia
k = SqExponentialKernel()
L = EvaluationFunctional(X)
op = 2.0 * Identity() + PartialDerivative((1,))   # 2·k + ∂k/∂x
fg = FunctionalGaussian(GP(k); u = L ∘ op)
```

# Caveats
- [`LinearDifferentialOperator`](@ref) currently only accepts (scaled)
  partial derivatives as summands. To combine `Identity` with partial
  derivatives, build a plain [`SumLinearFunctionOperator`](@ref) with `+`.
"""
struct Identity <: AbstractLinearFunctionOperator end

Base.show(io::IO, ::Identity) = print(io, "I")

# Generic pass-through. Type-specific overrides for ZeroMean, KernelSum,
# ScaledKernel, LinearlyScaledKernel, and the PV crosscov hierarchy live in
# `composition/operator_kernel/identity.jl` — those argument types are only
# defined in later load stages, and explicit overrides are required there to
# resolve ambiguity against the abstract-operator dispatches the framework
# already provides for those types.
(::Identity)(x, args...; kwargs...) = x
