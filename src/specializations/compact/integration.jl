import NearestNeighbors: BallTree, inrange

"""
    radial_antiderivative(k::CompactPolynomialKernel, ::Val{1})

Compute the first radial antiderivative of a compact polynomial kernel. Used for
one-sided integration (integrate-evaluate operations). Returns a function that
evaluates the antiderivative at a normalized radial distance `r`, clamped to the
support `[0, 1]`.
"""
function radial_antiderivative(k::CompactPolynomialKernel, ::Val{1})
    poly_int = Polynomials.integrate(k.poly)
    return (r) -> poly_int(min(r, 1.0))
end

"""
    radial_antiderivative(k::CompactPolynomialKernel, ::Val{2})

Compute the second radial antiderivative of a compact polynomial kernel. Used for
two-sided integration (integrate-integrate operations). For `r > 1`, the function
extends linearly beyond the support boundary.
"""
function radial_antiderivative(k::CompactPolynomialKernel, ::Val{2})
    poly_int = Polynomials.integrate(k.poly)
    poly_int2 = Polynomials.integrate(poly_int)
    poly_int2_norm = poly_int2 - poly_int2.coeffs[1]
    return (r) -> (
        r > 1.0 ? poly_int2_norm(1.0) + (r - 1.0) * poly_int(1.0) : poly_int2_norm(r)
    )
end

"""
    kernel_integrate_integrate(::StationaryKernelTrait, k::CompactPolynomialKernel, domains1, domains2)

Compute a sparse covariance matrix for two-sided integration of a compact
polynomial kernel. Uses `BallTree` spatial indexing to efficiently find
potentially interacting domain pairs, exploiting the kernel's compact support.

Returns a `SparseMatrixCSC` where entry `[i,j]` is `Cov(∫_{domains1[i]} k, ∫_{domains2[j]} k)`.
"""
function kernel_integrate_integrate(::StationaryKernelTrait, k::CompactPolynomialKernel, domains1, domains2)
    anti = radial_antiderivative(k, Val(2))
    ℓ = k.lengthscales

    lower_bounds1 = map(domain -> domain.lower, domains1)
    upper_bounds1 = map(domain -> domain.upper, domains1)

    lower_bounds2 = map(domain -> domain.lower, domains2)
    upper_bounds2 = map(domain -> domain.upper, domains2)

    mids1 = @. (lower_bounds1 + upper_bounds1) / 2
    half_lengths1 = (upper_bounds1 .- lower_bounds1) / 2

    tree_lower2 = BallTree(convert(Matrix{Float64}, reshape(lower_bounds2, 1, :)))
    tree_upper2 = BallTree(convert(Matrix{Float64}, reshape(upper_bounds2, 1, :)))
    I::Vector{Int}, J::Vector{Int}, V::Vector{Float64} = [], [], []
    for i in eachindex(mids1)
        @inbounds neighbors_lower2 =
            inrange(tree_lower2, [mids1[i]], half_lengths1[i] + ℓ)
        @inbounds neighbors_upper2 =
            inrange(tree_upper2, [mids1[i]], half_lengths1[i] + ℓ)
        neighbors = neighbors_lower2 ∪ neighbors_upper2
        for j in neighbors
            # radial_covfunc_two_sided_integral
            a, b = lower_bounds1[i], upper_bounds1[i]
            c, d = lower_bounds2[j], upper_bounds2[j]
            K = (x, y) -> anti(abs(x - y) / ℓ)
            @inbounds val = ℓ^2 * (K(b, c) - K(a, c) - K(b, d) + K(a, d))
            push!(I, i)
            push!(J, j)
            push!(V, val)
        end
    end
    return sparse(I, J, V, Base.length(lower_bounds1), Base.length(lower_bounds2))
end

"""
    kernel_integrate_evaluate(::StationaryKernelTrait, k::CompactPolynomialKernel, domains, X)

Compute a sparse covariance matrix for one-sided integration of a compact
polynomial kernel (integrate over domains, evaluate at points). Uses `BallTree`
spatial indexing for efficient neighbor search.

Returns a `SparseMatrixCSC` where entry `[i,j]` is `Cov(∫_{domains[i]} k, k(X[j]))`.
"""
function kernel_integrate_evaluate(::StationaryKernelTrait, k::CompactPolynomialKernel, domains, X::AbstractVector)
    anti = radial_antiderivative(k, Val(1))
    ℓ = k.lengthscales

    lower_bounds = map(domain -> domain.lower, domains)
    upper_bounds = map(domain -> domain.upper, domains)

    mids = @. (lower_bounds + upper_bounds) / 2
    half_lengths = (upper_bounds .- lower_bounds) / 2

    tree = BallTree(convert(Matrix{Float64}, reshape(X, 1, :)))
    I::Vector{Int}, J::Vector{Int}, V::Vector{Float64} = [], [], []
    for i in eachindex(mids)
        @inbounds neighbors = inrange(tree, [mids[i]], half_lengths[i] + ℓ)
        for j in neighbors
            # radial_covfunc_one_sided_integral
            a, b, x = lower_bounds[i], upper_bounds[i], X[j]
            a_sgn = (a < x) ? -1 : 1
            b_sgn = (b < x) ? -1 : 1
            r_a = abs(x - a) / ℓ
            r_b = abs(x - b) / ℓ
            @inbounds val = ℓ * (b_sgn * anti(r_b) - a_sgn * anti(r_a))
            push!(I, i)
            push!(J, j)
            push!(V, val)
        end
    end
    return sparse(I, J, V, Base.length(mids), Base.length(X))
end
