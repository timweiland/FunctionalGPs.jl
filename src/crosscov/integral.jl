using KernelFunctions: Kernel

export IntegralPVCrosscov

"""
    IntegralPVCrosscov{IntegralArg, TK, TD}

PV crosscov representing integration over domains applied to one argument of a
kernel. This is the generic type that replaces the kernel-specific integral
PV crosscov types (e.g., Matern1D_Identity_LebesgueIntegral).

The `IntegralArg` type parameter (1 or 2) indicates which kernel argument the
integration is applied to.

# Fields
- `integral_arg::Int`: Which argument (1 or 2) the integration is applied to
- `k::TK`: The kernel
- `domains::TD`: The collection of domains to integrate over

# Examples
```julia
julia> k = HalfIntegerMaternKernel(1, [0.8]);
julia> domains = [Interval(0.0, 0.3), Interval(0.3, 0.7)];
julia> ℒ = VectorizedLebesgueIntegral(domains);
julia> pv = ℒ(k, arg=1);  # Integrate first argument
julia> typeof(pv)
IntegralPVCrosscov{1, ...}

julia> # Apply another functional to build covariance matrix
julia> X = [0.0, 0.5, 1.0];
julia> ℒ_eval = EvaluationFunctional(X);
julia> K = ℒ_eval(pv);  # Returns a (2, 3) matrix
```

# See also
- [`EvaluationPVCrosscov`](@ref): For evaluation functionals
- [`kernel_integrate_integrate`](@ref): For integral × integral matrices
- [`kernel_integrate_evaluate`](@ref): For integral × evaluation matrices
"""
struct IntegralPVCrosscov{IntegralArg, TK <: Kernel, TD} <: ProcessVectorCrossCovariance
    integral_arg::Int
    k::TK
    domains::TD
end

function IntegralPVCrosscov(k::Kernel, domains, integral_arg::Integer)
    arg = Int(integral_arg)
    @assert arg ∈ (1, 2) "integral_arg must be 1 or 2, got $arg"
    return IntegralPVCrosscov{arg, typeof(k), typeof(domains)}(arg, k, domains)
end

# Accessor methods
randvar_batch_size(pv::IntegralPVCrosscov) = size(pv.domains)
randvar_arg(::IntegralPVCrosscov{IntegralArg}) where {IntegralArg} = IntegralArg

# Convenience accessors matching old interface
kernel(pv::IntegralPVCrosscov) = pv.k
covfunc(pv::IntegralPVCrosscov) = pv.k  # Alias for compatibility
domains(pv::IntegralPVCrosscov) = pv.domains

# Equality methods
function Base.isequal(pv1::IntegralPVCrosscov, pv2::IntegralPVCrosscov)
    return pv1.k == pv2.k && pv1.domains == pv2.domains && pv1.integral_arg == pv2.integral_arg
end

function Base.isapprox(pv1::IntegralPVCrosscov, pv2::IntegralPVCrosscov)
    return pv1.k ≈ pv2.k && pv1.domains ≈ pv2.domains && pv1.integral_arg == pv2.integral_arg
end

# kernelmatrix methods
function kernelmatrix(pv::IntegralPVCrosscov{1}, X::AbstractVector)
    # Integral on first argument, evaluate on second argument
    return kernel_integrate_evaluate(pv.k, pv.domains, X)
end

function kernelmatrix(pv::IntegralPVCrosscov{2}, X::AbstractVector)
    # Integral on second argument, evaluate on first argument
    result = kernel_integrate_evaluate(pv.k, pv.domains, X)
    return result'
end
