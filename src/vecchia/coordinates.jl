export get_coordinates

"""
    get_coordinates(L::AbstractLinearFunctional) -> Matrix{Float64}

Extract a `(d × n)` matrix of representative spatial coordinates from a linear
functional, where `d` is the spatial dimension and `n` is the number of
outputs. Used by `vecchia` to build a maximin ordering.

For evaluation functionals the coordinates are the evaluation points. For
integral functionals the centroid of each domain is used as a representative.
Composed functionals delegate to their inner functional. Tensor products take
the Cartesian product of the per-factor 1D coordinates.

For unrecognised types the fallback heuristic checks for `.X` and `.domains`
fields; otherwise it errors. Add a method specialised on your type to
participate in `vecchia`.
"""
function get_coordinates end

# --- Evaluation ----------------------------------------------------------------

get_coordinates(L::EvaluationFunctional) = _coords_from_points(L.X)

_coords_from_points(X::AbstractVector{<:Number}) = reshape(Float64.(collect(X)), 1, length(X))
function _coords_from_points(X::AbstractVector)
    n = length(X)
    first_x = first(X)
    d = first_x isa Number ? 1 : length(first_x)
    coords = Matrix{Float64}(undef, d, n)
    for (i, x) in enumerate(X)
        coords[:, i] .= x
    end
    return coords
end
_coords_from_points(X::AbstractRange) = reshape(collect(Float64, X), 1, length(X))
function _coords_from_points(X::FactorizedGrid)
    ranges = X.ranges
    d = length(ranges)
    n = prod(length, ranges)
    coords = Matrix{Float64}(undef, d, n)
    idx = 1
    for pt in Iterators.product(ranges...)
        for dim in 1:d
            coords[dim, idx] = pt[dim]
        end
        idx += 1
    end
    return coords
end

# --- Integral ------------------------------------------------------------------

get_coordinates(L::VectorizedLebesgueIntegral) = _coords_from_domains(L.domains)

_midpoint(iv::Interval) = (iv.lower + iv.upper) / 2
_midpoint(b::BoxDomain) = [(lo + hi) / 2 for (lo, hi) in b.bounds]

function _coords_from_domains(domains::AbstractVector{<:Interval})
    return reshape([Float64(_midpoint(d)) for d in domains], 1, length(domains))
end
function _coords_from_domains(domains::AbstractArray{<:BoxDomain})
    flat = vec(domains)
    n = length(flat)
    d = length(first(flat).bounds)
    coords = Matrix{Float64}(undef, d, n)
    for (i, dom) in enumerate(flat)
        coords[:, i] .= _midpoint(dom)
    end
    return coords
end
function _coords_from_domains(domains::FactorizedBoxDomains)
    interval_vecs = domains.interval_vecs
    d = length(interval_vecs)
    n = prod(length, interval_vecs)
    coords = Matrix{Float64}(undef, d, n)
    idx = 1
    for cell in Iterators.product(interval_vecs...)
        for dim in 1:d
            coords[dim, idx] = _midpoint(cell[dim])
        end
        idx += 1
    end
    return coords
end

# --- Composition / sum / tensor product ----------------------------------------

get_coordinates(L::LinFctlLinFuncOpConcat) = get_coordinates(L.linfctl)
get_coordinates(L::SumLinearFunctional) = get_coordinates(first(summands(L)))

function get_coordinates(L::TensorProductFunctional)
    factors = L.factors
    d = length(factors)
    factor_coords = map(_factor_coords_1d, factors)
    n = prod(length, factor_coords)
    coords = Matrix{Float64}(undef, d, n)
    idx = 1
    for pt in Iterators.product(factor_coords...)
        for dim in 1:d
            coords[dim, idx] = pt[dim]
        end
        idx += 1
    end
    return coords
end

# Per-factor 1D coordinates for a tensor product.
function _factor_coords_1d(L::EvaluationFunctional)
    X = L.X
    return X isa AbstractRange ? collect(Float64, X) :
        Float64[x isa Number ? x : x[1] for x in X]
end
_factor_coords_1d(L::VectorizedLebesgueIntegral) = Float64[_midpoint(d) for d in L.domains]

# --- Fallback ------------------------------------------------------------------

function get_coordinates(L::AbstractLinearFunctional)
    if hasproperty(L, :X)
        return _coords_from_points(L.X)
    elseif hasproperty(L, :domains)
        return _coords_from_domains(L.domains)
    end
    return error(
        "No `get_coordinates` method for $(typeof(L)). Define one to participate " *
            "in `vecchia` ordering."
    )
end
