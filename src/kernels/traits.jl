import KernelFunctions: Kernel, kernelmatrix, KernelTensorProduct

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

# KernelTensorProduct with FactorizedGrid: Kronecker product with per-factor trait dispatch.
# Each factor kernel goes through kernel_evaluate_evaluate individually, so stationary
# kernels on uniform ranges produce Toeplitz factors instead of dense matrices.
function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X::FactorizedGrid,
    )
    @assert length(k.kernels) == length(X.ranges)
    Ks = (kernel_evaluate_evaluate(k.kernels[i], X.ranges[i]) for i in 1:length(X.ranges))
    return reduce(kronecker, reverse(Tuple(Ks)))
end

function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X_left::FactorizedGrid,
        X_right::FactorizedGrid,
    )
    @assert length(X_left.ranges) == length(X_right.ranges)
    @assert length(k.kernels) == length(X_left.ranges)
    Ks = (
        kernel_evaluate_evaluate(k.kernels[i], X_left.ranges[i], X_right.ranges[i])
            for i in 1:length(X_left.ranges)
    )
    return reduce(kronecker, reverse(Tuple(Ks)))
end

# KernelTensorProduct with vector-of-vectors: Hadamard product of per-dimension
# lazy kernel matrices. Each factor goes through kernel_evaluate_evaluate for trait
# dispatch, so stationary kernels produce lazy representations.
function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X::AbstractVector{<:AbstractVector},
    )
    N = length(k.kernels)
    coords = ntuple(d -> [xi[d] for xi in X], N)
    Ks = ntuple(d -> kernel_evaluate_evaluate(k.kernels[d], coords[d]), N)
    return BroadcastArray(*, Ks...)
end

function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X_left::AbstractVector{<:AbstractVector},
        X_right::AbstractVector{<:AbstractVector},
    )
    N = length(k.kernels)
    coords_left = ntuple(d -> [xi[d] for xi in X_left], N)
    coords_right = ntuple(d -> [xi[d] for xi in X_right], N)
    Ks = ntuple(
        d -> kernel_evaluate_evaluate(k.kernels[d], coords_left[d], coords_right[d]), N,
    )
    return BroadcastArray(*, Ks...)
end

# Mixed: FactorizedGrid on one side, unstructured points on the other → KhatriRao.
# Rows = X_left, Cols = X_right. Grid side decomposes into per-dimension factors.
function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X_left::FactorizedGrid,
        X_right::AbstractVector{<:AbstractVector},
    )
    N = length(k.kernels)
    coords = ntuple(d -> [xi[d] for xi in X_right], N)
    Ks = [kernel_evaluate_evaluate(k.kernels[d], X_left.ranges[d], coords[d]) for d in 1:N]
    return KhatriRaoMatrix{1}(Ks)
end

function kernel_evaluate_evaluate(
        ::KernelStructureTrait,
        k::KernelTensorProduct,
        X_left::AbstractVector{<:AbstractVector},
        X_right::FactorizedGrid,
    )
    N = length(k.kernels)
    coords = ntuple(d -> [xi[d] for xi in X_left], N)
    Ks = [kernel_evaluate_evaluate(k.kernels[d], coords[d], X_right.ranges[d]) for d in 1:N]
    return KhatriRaoMatrix{2}(Ks)
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
