using EllipsisNotation

export Heat1DIBVPTruncatedSineICDirichletBC,
    domain, lindiffops, ic_fn, sample_ic, bc_fn, sample_bc

struct Heat1DIBVPTruncatedSineICDirichletBC <: IBVP
    domain::BoxDomain
    c::Float64
    ρ::Float64
    κ::Float64
    ic_coeffs::AbstractVector{Float64}
end

function Heat1DIBVPTruncatedSineICDirichletBC(
        domain::BoxDomain;
        c::Float64,
        ρ::Float64,
        κ::Float64,
        ic_coeffs::AbstractVector{<:Real},
    )
    return Heat1DIBVPTruncatedSineICDirichletBC(domain, c, ρ, κ, ic_coeffs)
end

domain(p::Heat1DIBVPTruncatedSineICDirichletBC) = p.domain
function lindiffops(p::Heat1DIBVPTruncatedSineICDirichletBC)
    return (
        LinearDifferentialOperator{1}(
            Dict(1 => Dict((1, 0) => p.c * p.ρ, (0, 2) => -p.κ)),
        ),
    )
end
function ic_fn(p::Heat1DIBVPTruncatedSineICDirichletBC, x)
    return sum(
        map(1:length(p.ic_coeffs)) do i
            return p.ic_coeffs[i] * sin.(i * π .* x)
        end
    )
end
function sample_ic(
        p::Heat1DIBVPTruncatedSineICDirichletBC,
        N::Int;
        noise::Union{Real, Nothing} = 1.0e-8,
    )
    X_ic = FactorizedGrid(
        [0.0],
        convert(Vector, range(p.domain[2, 1]; stop = p.domain[2, 2], length = N)),
    )
    Xs = convert(Array, X_ic)[1, :, 2]
    Y_ic = ic_fn(p, Xs)
    return LinearObservation(EvaluationFunctional(X_ic), Y_ic; noise = noise)
end
bc_fn(::Heat1DIBVPTruncatedSineICDirichletBC, x) = spzeros(size(x)[1:(end - 1)]...)
function sample_bc(
        p::Heat1DIBVPTruncatedSineICDirichletBC,
        N::Int;
        noise::Union{Real, Nothing} = 1.0e-8,
    )
    X_bc = FactorizedGrid(
        convert(Vector, range(p.domain[1, 1]; stop = p.domain[1, 2], length = N)),
        [0.0, 1.0],
    )
    Xs = convert(Array, X_bc)
    Y_bc = bc_fn(p, Xs)
    return LinearObservation(EvaluationFunctional(X_bc), Y_bc; noise = noise)
end
function solution(p::Heat1DIBVPTruncatedSineICDirichletBC)
    α = p.κ / (p.c * p.ρ)
    half_angular_frequencies =
        (π / (p.domain[2, 2] - p.domain[2, 1])) * (1:length(p.ic_coeffs))
    decay_rates = α * (half_angular_frequencies .^ 2)
    # From separation of variables
    function fn(T::AbstractArray, X::AbstractArray)
        return sum(
            map(1:length(p.ic_coeffs)) do i
                return p.ic_coeffs[i] .* exp.(-decay_rates[i] .* (T .- p.domain[1, 1])) .*
                    sin.(half_angular_frequencies[i] .* (X .- p.domain[2, 1]))
            end,
        )
    end
    fn(t::Real, x::Real) = fn([t], [x])[1]
    function fn(X::AbstractArray)
        @assert size(X)[end] == 2
        return fn(X[.., 1], X[.., 2])
    end
    return fn
end
