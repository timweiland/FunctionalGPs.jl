import KernelFunctions: Kernel, kernelmatrix, KernelTensorProduct, ColVecs

export KernelStructureTrait,
    GenericKernelTrait,
    StationaryKernelTrait,
    SignedStationaryKernelTrait,
    kernel_structure,
    kernel_evaluate_evaluate,
    kernel_integrate_integrate,
    kernel_integrate_evaluate

"""
    KernelStructureTrait

Trait describing structural properties of a kernel that influence how linear
functionals should assemble covariance matrices. Subtypes enable specialised
lazy matrix representations.
"""
abstract type KernelStructureTrait end

"""
    GenericKernelTrait()

Fallback trait used when a kernel does not expose special structure.
"""
struct GenericKernelTrait <: KernelStructureTrait end

"""
    StationaryKernelTrait()

Trait indicating that the kernel is stationary and supports fast evaluation via
`StationaryKernelMatrix` when both sides are evaluation functionals with point
coordinates.
"""
struct StationaryKernelTrait <: KernelStructureTrait end

"""
    SignedStationaryKernelTrait()

Trait for kernels that are signed-stationary (e.g., kernels involving derivatives
with odd total order). These kernels have structure k(x,y) = sign(x-y) * f(|x-y|)
and support specialized lazy matrix representations.
"""
struct SignedStationaryKernelTrait <: KernelStructureTrait end

"""
    kernel_structure(k)

Return the structural trait associated with kernel `k`. The default returns
`GenericKernelTrait()`; specific kernels override this to signal richer
structure.

# Examples
```julia
julia> k = HalfIntegerMaternKernel(2, [1.0]);
julia> kernel_structure(k)
StationaryKernelTrait()

julia> k_compact = CompactPolynomialKernel(Polynomial([1.0, -2.0]));
julia> kernel_structure(k_compact)
StationaryKernelTrait()
```
"""
kernel_structure(::Kernel) = GenericKernelTrait()

"""
    kernel_evaluate_evaluate(k, X)
    kernel_evaluate_evaluate(k, X_left, X_right)

Assemble the covariance matrix resulting from applying evaluation functionals
at coordinate collections `X` (and optionally `X_left`, `X_right`) to kernel
`k`. The return value is a subtype of `CovarianceMatrix` or a fully materialised
matrix when no lazy representation is available.

# Examples
```julia
julia> k = HalfIntegerMaternKernel(2, [1.0]);
julia> X = collect(range(0, 1, length=10));
julia> K = kernel_evaluate_evaluate(k, X);  # Lazy Toeplitz matrix
julia> size(K)
(10, 10)

julia> X_left = [0.0, 0.5];
julia> X_right = [0.2, 0.7, 0.9];
julia> K_cross = kernel_evaluate_evaluate(k, X_left, X_right);
julia> size(K_cross)
(2, 3)
```
"""
function kernel_evaluate_evaluate(k::Kernel, X)
    return kernel_evaluate_evaluate(kernel_structure(k), k, X)
end

function kernel_evaluate_evaluate(k::Kernel, X_left, X_right)
    return kernel_evaluate_evaluate(kernel_structure(k), k, X_left, X_right)
end

kernel_evaluate_evaluate(::KernelStructureTrait, k::Kernel, X) = kernelmatrix(k, X)

kernel_evaluate_evaluate(::KernelStructureTrait, k::Kernel, X_left, X_right) =
    kernelmatrix(k, X_left, X_right)

# KernelTensorProduct requires ColVecs wrapper for vector-of-vectors input
function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X::AbstractVector{<:AbstractVector},
    )
    X_mat = reduce(hcat, X)  # Convert to matrix (d × n)
    return kernelmatrix(k, ColVecs(X_mat))
end

function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X_left::AbstractVector{<:AbstractVector},
        X_right::AbstractVector{<:AbstractVector},
    )
    X_left_mat = reduce(hcat, X_left)
    X_right_mat = reduce(hcat, X_right)
    return kernelmatrix(k, ColVecs(X_left_mat), ColVecs(X_right_mat))
end

"""
    kernel_integrate_integrate(k, domains)
    kernel_integrate_integrate(k, domains1, domains2)

Assemble the covariance matrix resulting from applying integration functionals
over domain collections `domains` (and optionally `domains1`, `domains2`) to
kernel `k`. The return value is a lazy matrix representation when the kernel
has exploitable structure (e.g., stationary), or a fully materialised matrix
otherwise.

The single-argument version applies integration to both sides with the same
domains (symmetric case). Implementations may specialize this for efficiency.

# Examples
```julia
julia> k = HalfIntegerMaternKernel(1, [0.8]);
julia> domains = [Interval(0.0, 0.2), Interval(0.2, 0.5), Interval(0.5, 1.0)];
julia> K = kernel_integrate_integrate(k, domains);  # Lazy Toeplitz-like matrix
julia> size(K)
(3, 3)

julia> domains_left = [Interval(0.0, 0.3), Interval(0.3, 0.6)];
julia> domains_right = [Interval(0.0, 0.25), Interval(0.25, 0.5), Interval(0.5, 1.0)];
julia> K_cross = kernel_integrate_integrate(k, domains_left, domains_right);
julia> size(K_cross)
(2, 3)
```
"""
function kernel_integrate_integrate(k::Kernel, domains)
    return kernel_integrate_integrate(kernel_structure(k), k, domains)
end

function kernel_integrate_integrate(k::Kernel, domains1, domains2)
    return kernel_integrate_integrate(kernel_structure(k), k, domains1, domains2)
end

# Default: single-domain version calls two-domain version with same domains
function kernel_integrate_integrate(trait::KernelStructureTrait, k::Kernel, domains)
    return kernel_integrate_integrate(trait, k, domains, domains)
end

# Generic fallback for two-domain case - not implemented
function kernel_integrate_integrate(trait::KernelStructureTrait, k::Kernel, domains1, domains2)
    error(
        "kernel_integrate_integrate not implemented for kernel type $(typeof(k)). " *
            "This requires either a stationary kernel trait or a specialized implementation."
    )
end

"""
    kernel_integrate_evaluate(k, domains, points)

Assemble the covariance matrix resulting from applying integration functionals
over `domains` to one argument and evaluation functionals at `points` to the
other argument of kernel `k`. The return value is a lazy matrix representation
when available, or a fully materialised matrix otherwise.

# Examples
```julia
julia> k = HalfIntegerMaternKernel(2, [1.3]);
julia> domains = [Interval(0.0, 0.4), Interval(0.4, 0.9)];
julia> X = collect(range(0.0, 1.0, length=5));
julia> K = kernel_integrate_evaluate(k, domains, X);
julia> size(K)
(2, 5)
```

# Notes
The resulting matrix K[i,j] represents cov(∫ over domains[i], evaluate at X[j]).
"""
function kernel_integrate_evaluate(k::Kernel, domains, points)
    return kernel_integrate_evaluate(kernel_structure(k), k, domains, points)
end

# Generic fallback - not implemented
function kernel_integrate_evaluate(trait::KernelStructureTrait, k::Kernel, domains, points)
    error(
        "kernel_integrate_evaluate not implemented for kernel type $(typeof(k)). " *
            "This requires either a stationary kernel trait or a specialized implementation."
    )
end
