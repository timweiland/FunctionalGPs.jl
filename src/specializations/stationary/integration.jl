"""
    _interval_bounds_matrices(domains, ::Type{T})

Extract lower and upper bounds from a collection of interval domains into
separate `(n, 1)` matrices of type `T`. Used internally for constructing
integration matrices.
"""
function _interval_bounds_matrices(domains, ::Type{T}) where {T <: AbstractFloat}
    n = length(domains)
    lowers = Matrix{T}(undef, n, 1)
    uppers = Matrix{T}(undef, n, 1)
    @inbounds for i in 1:n
        d = domains[i]
        lowers[i, 1] = T(d.lower)
        uppers[i, 1] = T(d.upper)
    end
    return lowers, uppers
end

_extract_lengthscale(ℓ) = only(ℓ)

# Dispatchable accessor for kernel lengthscales
# Override this for kernels that don't have a .lengthscales field
_kernel_lengthscales(k) = k.lengthscales

"""
    kernel_integrate_integrate(::StationaryKernelTrait, k::Kernel, domains1, domains2)

Compute the covariance matrix for two-sided integration of a stationary kernel
over 1D interval domains. Uses the kernel's second radial antiderivative
(`radial_antiderivative(k, Val(2))`) to construct a lazy matrix as a sum of
`StationaryKernelMatrix` terms.

The result represents `Cov(∫_{domains1[i]} k, ∫_{domains2[j]} k)`.
"""
function kernel_integrate_integrate(::StationaryKernelTrait, k::Kernel, domains1, domains2)
    # Validation
    ℓ_val = _extract_lengthscale(_kernel_lengthscales(k))
    ℓ_val isa Number || error("Stationary 1D integration requires scalar lengthscale")

    # Type promotion
    base_type = promote_type(
        typeof(domains1[1].lower),
        typeof(domains1[1].upper),
        typeof(domains2[1].lower),
        typeof(domains2[1].upper),
    )
    T = float(base_type)

    # Get radial antiderivative for two-sided integration
    anti = radial_antiderivative(k, Val(2))

    # Build lazy matrix from interval bounds
    lowers1, uppers1 = _interval_bounds_matrices(domains1, T)
    lowers2, uppers2 = _interval_bounds_matrices(domains2, T)

    ℓT = T(ℓ_val)
    radial_map = let anti = anti, ℓ2 = ℓT^2, zero_T = zero(T)
        r2 -> begin
            τ = sqrt(max(r2, zero_T))
            return ℓ2 * anti(τ)
        end
    end

    scales = T[inv(ℓT)]

    # Construct as sum of 4 stationary kernel matrices
    term_bc = StationaryKernelMatrix(uppers1, lowers2, radial_map; scales = scales)
    term_ac = -1.0 * StationaryKernelMatrix(lowers1, lowers2, radial_map; scales = scales)
    term_bd = -1.0 * StationaryKernelMatrix(uppers1, uppers2, radial_map; scales = scales)
    term_ad = StationaryKernelMatrix(lowers1, uppers2, radial_map; scales = scales)

    return ApplyArray(+, term_bc, term_ac, term_bd, term_ad)
end

"""
    kernel_integrate_integrate(::StationaryKernelTrait, k::Kernel, domains)

Symmetric version of two-sided integration where both sides use the same domains.
Delegates to the two-argument form.
"""
function kernel_integrate_integrate(::StationaryKernelTrait, k::Kernel, domains)
    # Delegate to two-domain version
    return kernel_integrate_integrate(StationaryKernelTrait(), k, domains, domains)
end

"""
    kernel_integrate_evaluate(::StationaryKernelTrait, k::Kernel, domains, X)

Compute the covariance matrix for one-sided integration (integrate over domains,
evaluate at points). Uses the kernel's first radial antiderivative
(`radial_antiderivative(k, Val(1))`) and returns a lazy matrix via
`SignedStationaryKernelMatrix`.

The result represents `Cov(∫_{domains[i]} k, k(X[j]))`.
"""
function kernel_integrate_evaluate(::StationaryKernelTrait, k::Kernel, domains, X::AbstractVector)
    isempty(domains) && return Matrix{Float64}(undef, 0, length(X))

    coords = _stationary_coordinates(X)
    coords === nothing && error("Cannot convert points to stationary coordinates")

    base_type = promote_type(eltype(X), typeof(domains[1].lower), typeof(domains[1].upper))
    T = float(base_type)
    T <: AbstractFloat || error("Require float type for integration")

    ℓ_val = _extract_lengthscale(_kernel_lengthscales(k))
    ℓ_val isa Number || error("Require scalar lengthscale for 1D integration")

    # Get radial antiderivative for one-sided integration
    anti = radial_antiderivative(k, Val(1))

    # Build lazy matrix from interval bounds and points
    lowers, uppers = _interval_bounds_matrices(domains, T)
    points = reshape(T.(X), :, 1)

    ℓT = T(ℓ_val)
    signed_map = let anti = anti, ℓT = ℓT, zero_T = zero(T), one_T = one(T)
        (r2, s) -> begin
            τ = sqrt(max(r2, zero_T))
            s_eff = iszero(s) ? one_T : s
            return s_eff * ℓT * anti(τ)
        end
    end

    scales = T[inv(ℓT)]
    upper_matrix = SignedStationaryKernelMatrix(uppers, points, signed_map; scales = scales)
    lower_matrix = SignedStationaryKernelMatrix(lowers, points, signed_map; scales = scales)

    return upper_matrix - lower_matrix
end
