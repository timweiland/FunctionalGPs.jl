export CompactPolynomialCovFunc1D_Identity_LebesgueIntegral, integrate
using NearestNeighbors: BallTree, inrange

struct CompactPolynomialCovFunc1D_Identity_LebesgueIntegral <:
    RadialCovarianceFunction1D_Identity_LebesgueIntegral
    covfunc::CompactPolynomialKernel
    domains::AbstractVector{Interval{Float64}}
    randvar_arg::Int
end

function kernelmatrix(
        pv::CompactPolynomialCovFunc1D_Identity_LebesgueIntegral,
        X::AbstractVector,
    )
    anti = radial_antiderivative(pv.covfunc, Val(1))
    ℓ = pv.covfunc.lengthscales

    lower_bounds = map(domain -> domain.lower, domains(pv))
    upper_bounds = map(domain -> domain.upper, domains(pv))

    mids = @. (lower_bounds + upper_bounds) / 2
    half_lengths = (upper_bounds .- lower_bounds) / 2

    tree = BallTree(convert(Matrix{Float64}, reshape(X, 1, :)))
    I::Vector{Int}, J::Vector{Int}, V::Vector{Float64} = [], [], []
    for i in eachindex(mids)
        @inbounds neighbors = inrange(tree, [mids[i]], half_lengths[i] + ℓ)
        for j in neighbors
            @inbounds val = radial_covfunc_one_sided_integral(
                lower_bounds[i],
                upper_bounds[i],
                X[j];
                anti_derivative = anti,
                ℓ = ℓ,
            )
            push!(I, i)
            push!(J, j)
            push!(V, val)
        end
    end
    res = sparse(I, J, V, Base.length(mids), Base.length(X))
    if pv.randvar_arg == 2
        res = res'
    end
    return res
end

function integrate(
        k::CompactPolynomialKernel,
        domains1,
        domains2,
    )
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
            @inbounds val = radial_covfunc_two_sided_integral(
                lower_bounds1[i],
                upper_bounds1[i],
                lower_bounds2[j],
                upper_bounds2[j];
                anti_derivative = anti,
                ℓ = ℓ,
            )
            push!(I, i)
            push!(J, j)
            push!(V, val)
        end
    end
    return sparse(I, J, V, Base.length(lower_bounds1), Base.length(lower_bounds2))
end
