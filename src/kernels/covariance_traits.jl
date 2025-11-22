import KernelFunctions: Kernel, kernelmatrix

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

struct SignedStationaryKernelTrait <: KernelStructureTrait end

"""
    kernel_structure(k)

Return the structural trait associated with kernel `k`. The default returns
`GenericKernelTrait()`; specific kernels override this to signal richer
structure.
"""
kernel_structure(::Kernel) = GenericKernelTrait()

"""
    kernel_evaluate_evaluate(k, X)
    kernel_evaluate_evaluate(k, X_left, X_right)

Assemble the covariance matrix resulting from applying evaluation functionals
at coordinate collections `X` (and optionally `X_left`, `X_right`) to kernel
`k`. The return value is a subtype of `CovarianceMatrix` or a fully materialised
matrix when no lazy representation is available.
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
