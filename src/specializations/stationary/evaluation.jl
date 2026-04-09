import KernelFunctions: kernelmatrix
import ToeplitzMatrices: SymmetricToeplitz

"""
    kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X::AbstractRange)

Construct a symmetric Toeplitz covariance matrix for a stationary kernel evaluated
on a uniformly spaced range. Exploits the constant spacing to avoid redundant
kernel evaluations.

Returns a `SymmetricToeplitz` matrix when the kernel supports stationary
specialization; otherwise falls back to `kernelmatrix`.
"""
function kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X::AbstractRange)
    first_point = first(X)
    step = Base.step(X)
    len = Base.length(X)
    T = float(promote_type(typeof(first_point), typeof(step)))
    T <: Real || return kernelmatrix(k, X)
    spec = stationary_kernel_spec(k, T)
    spec === nothing && return kernelmatrix(k, X)
    length(spec.scales) == 1 || return kernelmatrix(k, X)
    # Promote T to include scale type (may be AD dual numbers)
    S = promote_type(T, eltype(spec.scales))
    scaled_step = S(step) * spec.scales[1]
    coeffs = Vector{S}(undef, len)
    for idx in 1:len
        dist = scaled_step * (idx - 1)
        coeffs[idx] = spec.radial_map(dist^2)
    end
    return SymmetricToeplitz(coeffs)
end

"""
    kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X)

Construct a lazy `StationaryKernelMatrix` for a stationary kernel evaluated at
arbitrary point coordinates. The matrix computes entries on-demand using scaled
squared distances.

Falls back to `kernelmatrix` if the input cannot be converted to stationary
coordinates or the kernel does not provide a stationary specification.
"""
function kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X)
    coords = _stationary_coordinates(X)
    coords === nothing && return kernelmatrix(k, X)
    T = eltype(coords)
    T <: Real || return kernelmatrix(k, X)
    spec = stationary_kernel_spec(k, T)
    spec === nothing && return kernelmatrix(k, X)
    return StationaryKernelMatrix(coords, spec.radial_map; scales = spec.scales)
end

"""
    kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X_left, X_right)

Construct a lazy `StationaryKernelMatrix` representing the cross-covariance between
two sets of points for a stationary kernel. Uses the kernel's radial map on scaled
squared distances.
"""
function kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X_left, X_right)
    coords_left = _stationary_coordinates(X_left)
    coords_right = _stationary_coordinates(X_right)
    if coords_left === nothing || coords_right === nothing
        return kernelmatrix(k, X_left, X_right)
    end
    T_left = eltype(coords_left)
    T_right = eltype(coords_right)
    if !(T_left <: Real && T_right <: Real)
        return kernelmatrix(k, X_left, X_right)
    end
    if T_left != T_right
        return kernelmatrix(k, X_left, X_right)
    end
    spec = stationary_kernel_spec(k, T_left)
    spec === nothing && return kernelmatrix(k, X_left, X_right)
    return StationaryKernelMatrix(coords_left, coords_right, spec.radial_map; scales = spec.scales)
end

"""
    kernel_evaluate_evaluate(::SignedStationaryKernelTrait, k::Kernel, X)

Construct a lazy `SignedStationaryKernelMatrix` for kernels with signed-stationary
structure (e.g., odd-order derivative kernels). The kernel value depends on both
the squared distance and the sign of the coordinate difference.
"""
function kernel_evaluate_evaluate(::SignedStationaryKernelTrait, k::Kernel, X)
    coords = _stationary_coordinates(X)
    coords === nothing && return kernelmatrix(k, X)
    T = eltype(coords)
    T <: Real || return kernelmatrix(k, X)
    spec = stationary_kernel_spec(k, T)
    spec === nothing && return kernelmatrix(k, X)
    spec isa SignedStationaryKernelSpec || return kernelmatrix(k, X)
    return SignedStationaryKernelMatrix(coords, spec.signed_map; scales = spec.scales)
end

"""
    kernel_evaluate_evaluate(::SignedStationaryKernelTrait, k::Kernel, X_left, X_right)

Construct a lazy `SignedStationaryKernelMatrix` for cross-covariance between two
point sets using a signed-stationary kernel.
"""
function kernel_evaluate_evaluate(::SignedStationaryKernelTrait, k::Kernel, X_left, X_right)
    coords_left = _stationary_coordinates(X_left)
    coords_right = _stationary_coordinates(X_right)
    if coords_left === nothing || coords_right === nothing
        return kernelmatrix(k, X_left, X_right)
    end
    T_left = eltype(coords_left)
    T_right = eltype(coords_right)
    if !(T_left <: Real && T_right <: Real)
        return kernelmatrix(k, X_left, X_right)
    end
    if T_left != T_right
        return kernelmatrix(k, X_left, X_right)
    end
    spec = stationary_kernel_spec(k, T_left)
    spec === nothing && return kernelmatrix(k, X_left, X_right)
    spec isa SignedStationaryKernelSpec || return kernelmatrix(k, X_left, X_right)
    return SignedStationaryKernelMatrix(coords_left, coords_right, spec.signed_map; scales = spec.scales)
end
