module StationaryUtils

export _stationary_scale_values, _scaled_inputs_and_norms

"""
    _stationary_scale_values(::Type{T}, ncols, scales)

Return a vector of length `ncols` containing per-dimension scale factors of
type `T`. When `scales === nothing`, the vector is filled with ones; otherwise
the entries are copied and converted to `T`.
"""
function _stationary_scale_values(::Type{T}, ncols::Integer, scales) where {T <: AbstractFloat}
    values = Vector{T}(undef, ncols)
    if scales === nothing
        fill!(values, one(T))
    else
        if length(scales) != ncols
            throw(DimensionMismatch("scales length $(length(scales)) ≠ number of columns $(ncols)"))
        end
        values .= T.(scales)
    end
    return values
end

"""
    _scaled_inputs_and_norms(X, scale_values)

Scale the columns of `X` by `scale_values` and return both the scaled data and
their row-wise squared norms.
"""
function _scaled_inputs_and_norms(
        X::AbstractMatrix{T},
        scale_values::AbstractVector{T},
    ) where {T <: AbstractFloat}
    ncols = size(X, 2)
    scaled_inputs = similar(X)
    scale_vec = similar(X, T, ncols)
    copyto!(scale_vec, scale_values)
    reshape_scales = reshape(scale_vec, 1, ncols)
    @. scaled_inputs = X * reshape_scales
    norms = vec(sum(abs2, scaled_inputs; dims = 2))
    return scaled_inputs, norms
end

end
