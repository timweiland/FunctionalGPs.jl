# Stationary kernel specifications for efficient matrix computation
# (kernel evaluation methods are in kernels/matern.jl)

"""
    stationary_kernel_spec(k::HalfIntegerMaternKernel{P}, ::Type{T})

Return a `StationaryKernelSpec` for efficient lazy matrix construction with a
half-integer Matérn kernel. The radial map evaluates the exponential-polynomial
form at scaled distances.
"""
function stationary_kernel_spec(
        k::HalfIntegerMaternKernel{P},
        ::Type{T},
    ) where {P, T <: Real}
    sqrt_2nu = sqrt(T(2 * P + 1))
    # Don't convert lengthscales to T — they may carry AD dual numbers.
    # Normalise scalar lengthscales to a 1-element vector: a scalar lengthscale
    # is allowed by the constructor, but `collect` of a scalar yields a 0-dim
    # array, which the `StationaryKernelSpec` constructor (TS <: AbstractVector)
    # rejects. Broadcasting over the vector always returns a proper Vector.
    lengthscales = k.lengthscales isa AbstractVector ? k.lengthscales : [k.lengthscales]
    scales = sqrt_2nu ./ lengthscales
    radial_map = let kernel = k
        r2 -> begin
            r = _safe_dist(r2)
            return _exp_poly(kernel, r)
        end
    end
    return StationaryKernelSpec(scales, radial_map)
end

"""
    stationary_kernel_spec(k::HalfIntegerMaternDerivativeOddKernel, ::Type{T})

Return a `SignedStationaryKernelSpec` for odd-order derivatives of half-integer
Matérn kernels. The signed map accounts for the antisymmetric structure
introduced by odd total derivative order.
"""
function stationary_kernel_spec(
        k::HalfIntegerMaternDerivativeOddKernel,
        ::Type{T},
    ) where {T <: Real}
    base_spec = stationary_kernel_spec(k.base, T)
    base_spec === nothing && return nothing
    # Don't convert coefficient to T — it may carry AD dual numbers
    signed_map = let kernel = k, coeff = k.coefficient
        (r2, s) -> begin
            τ = _safe_dist(r2)
            return s * coeff * exp(-τ) * kernel.polynomial(τ)
        end
    end
    return SignedStationaryKernelSpec(base_spec.scales, signed_map)
end

"""
    stationary_kernel_spec(k::HalfIntegerMaternDerivativeEvenKernel, ::Type{T})

Return a `StationaryKernelSpec` for even-order derivatives of half-integer
Matérn kernels. Even total derivative order preserves the symmetric radial
structure.
"""
function stationary_kernel_spec(
        k::HalfIntegerMaternDerivativeEvenKernel,
        ::Type{T},
    ) where {T <: Real}
    base_spec = stationary_kernel_spec(k.base, T)
    base_spec === nothing && return nothing
    # Don't convert coefficient to T — it may carry AD dual numbers
    radial_map = let kernel = k, coeff = k.coefficient
        r2 -> begin
            τ = _safe_dist(r2)
            return coeff * exp(-τ) * kernel.polynomial(τ)
        end
    end
    return StationaryKernelSpec(base_spec.scales, radial_map)
end
