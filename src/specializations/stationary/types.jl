import KernelFunctions: Kernel

export radial_antiderivative, StationaryKernelSpec, SignedStationaryKernelSpec, stationary_kernel_spec

"""
    radial_antiderivative(k, ::Val{N})

Generic function for computing the Nth radial antiderivative of a kernel.
Specific kernel types implement this for N=1 (one-sided integration) and
N=2 (two-sided integration).
"""
function radial_antiderivative end

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

"""
    _safe_dist(r2)

AD-safe distance computation: clamps negative values and returns `zero(r2)`
when `r2` is zero, avoiding the NaN that ForwardDiff produces from `sqrt'(0) = Inf`.
"""
function _safe_dist(r2)
    r2 = max(r2, zero(r2))
    iszero(r2) && return zero(r2)
    return sqrt(r2)
end
