# Stationary kernel specifications for efficient matrix computation
# (kernel evaluation methods are in kernels/matern.jl)

function stationary_kernel_spec(
        k::HalfIntegerMaternKernel{P},
        ::Type{T},
    ) where {P, T <: AbstractFloat}
    sqrt_2nu = sqrt(T(2 * P + 1))
    scales = collect(sqrt_2nu ./ T.(k.lengthscales))
    radial_map = let kernel = k, zero_T = zero(T)
        r2 -> begin
            r = sqrt(max(r2, zero_T))
            return convert(T, _exp_poly(kernel, r))
        end
    end
    return StationaryKernelSpec(scales, radial_map)
end

function stationary_kernel_spec(
        k::HalfIntegerMaternDerivativeOddKernel,
        ::Type{T},
    ) where {T <: AbstractFloat}
    base_spec = stationary_kernel_spec(k.base, T)
    base_spec === nothing && return nothing
    coeff_T = T(k.coefficient)
    signed_map = let kernel = k, coeff = coeff_T, zero_T = zero(T)
        (r2, s) -> begin
            τ = sqrt(max(r2, zero_T))
            return s * coeff * exp(-τ) * kernel.polynomial(T(τ))
        end
    end
    return SignedStationaryKernelSpec(base_spec.scales, signed_map)
end

function stationary_kernel_spec(
        k::HalfIntegerMaternDerivativeEvenKernel,
        ::Type{T},
    ) where {T <: AbstractFloat}
    base_spec = stationary_kernel_spec(k.base, T)
    base_spec === nothing && return nothing
    coeff_T = T(k.coefficient)
    radial_map = let kernel = k, coeff = coeff_T, zero_T = zero(T)
        r2 -> begin
            τ = sqrt(max(r2, zero_T))
            value = coeff * exp(-τ) * kernel.polynomial(T(τ))
            return convert(T, value)
        end
    end
    return StationaryKernelSpec(base_spec.scales, radial_map)
end
