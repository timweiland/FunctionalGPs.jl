import KernelFunctions: Kernel, kernelmatrix
import ToeplitzMatrices: SymmetricToeplitz

"""
    StationaryKernelSpec(scales, radial_map)

Container describing how a stationary kernel should be assembled lazily. The
`scales` vector rescales coordinate columns, while `radial_map` maps squared
distances to covariance values.
"""
struct StationaryKernelSpec{TS <: AbstractVector, F}
    scales::TS
    radial_map::F
end

"""
    SignedStationaryKernelSpec(scales, signed_map)

Container for signed stationary kernels where the kernel depends on both
squared distance and the sign of the coordinate difference.
"""
struct SignedStationaryKernelSpec{TS <: AbstractVector, F}
    scales::TS
    signed_map::F
end

"""
    stationary_kernel_spec(k, ::Type{T})

Return a `StationaryKernelSpec` describing how to construct covariance matrices
for kernel `k` with element type `T`. The default returns `nothing`, signalling
that no stationary specialisation is available.
"""
stationary_kernel_spec(::Kernel, ::Type{T}) where {T} = nothing

"""
    _stationary_coordinates(points)

Convert coordinate collections into the `(n, d)` matrix form expected by
`StationaryKernelMatrix`. Real vectors are reshaped into single-column matrices;
other structures that already behave as matrices are passed through. Returns
`nothing` when conversion is not supported.
"""
_stationary_coordinates(points::AbstractMatrix) = points
function _stationary_coordinates(points::AbstractVector{T}) where {T <: Real}
    return reshape(points, :, 1)
end
_stationary_coordinates(::Any) = nothing

function kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X::AbstractRange)
    first_point = first(X)
    step = Base.step(X)
    len = Base.length(X)
    T = float(promote_type(typeof(first_point), typeof(step)))
    T <: AbstractFloat || return kernelmatrix(k, X)
    spec = stationary_kernel_spec(k, T)
    spec === nothing && return kernelmatrix(k, X)
    length(spec.scales) == 1 || return kernelmatrix(k, X)
    scaled_step = T(step) * T(spec.scales[1])
    offsets = collect(0:(len - 1))
    coeffs = Vector{T}(undef, len)
    for (idx, offset) in pairs(offsets)
        dist = scaled_step * T(offset)
        coeffs[idx] = spec.radial_map(dist^2)
    end
    return SymmetricToeplitz(coeffs)
end

function kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X)
    coords = _stationary_coordinates(X)
    coords === nothing && return kernelmatrix(k, X)
    T = eltype(coords)
    T <: AbstractFloat || return kernelmatrix(k, X)
    spec = stationary_kernel_spec(k, T)
    spec === nothing && return kernelmatrix(k, X)
    return StationaryKernelMatrix(coords, spec.radial_map; scales = spec.scales)
end

function kernel_evaluate_evaluate(::StationaryKernelTrait, k::Kernel, X_left, X_right)
    coords_left = _stationary_coordinates(X_left)
    coords_right = _stationary_coordinates(X_right)
    if coords_left === nothing || coords_right === nothing
        return kernelmatrix(k, X_left, X_right)
    end
    T_left = eltype(coords_left)
    T_right = eltype(coords_right)
    if !(T_left <: AbstractFloat && T_right <: AbstractFloat)
        return kernelmatrix(k, X_left, X_right)
    end
    if T_left != T_right
        return kernelmatrix(k, X_left, X_right)
    end
    spec = stationary_kernel_spec(k, T_left)
    spec === nothing && return kernelmatrix(k, X_left, X_right)
    return StationaryKernelMatrix(coords_left, coords_right, spec.radial_map; scales = spec.scales)
end

function kernel_evaluate_evaluate(::SignedStationaryKernelTrait, k::Kernel, X)
    coords = _stationary_coordinates(X)
    coords === nothing && return kernelmatrix(k, X)
    T = eltype(coords)
    T <: AbstractFloat || return kernelmatrix(k, X)
    spec = stationary_kernel_spec(k, T)
    spec === nothing && return kernelmatrix(k, X)
    spec isa SignedStationaryKernelSpec || return kernelmatrix(k, X)
    return SignedStationaryKernelMatrix(coords, spec.signed_map; scales = spec.scales)
end

function kernel_evaluate_evaluate(::SignedStationaryKernelTrait, k::Kernel, X_left, X_right)
    coords_left = _stationary_coordinates(X_left)
    coords_right = _stationary_coordinates(X_right)
    if coords_left === nothing || coords_right === nothing
        return kernelmatrix(k, X_left, X_right)
    end
    T_left = eltype(coords_left)
    T_right = eltype(coords_right)
    if !(T_left <: AbstractFloat && T_right <: AbstractFloat)
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
