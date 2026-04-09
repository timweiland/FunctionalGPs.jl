module StationaryUtils

export _stationary_scale_values, _scaled_inputs_and_norms

"""
    _stationary_scale_values(ncols, scales)

Return a vector of length `ncols` containing per-dimension scale factors.
When `scales === nothing`, the vector is filled with ones; otherwise
the entries are collected from `scales`.
"""
function _stationary_scale_values(ncols::Integer, scales)
    if scales === nothing
        return ones(ncols)
    end
    if length(scales) != ncols
        throw(DimensionMismatch("scales length $(length(scales)) ≠ number of columns $(ncols)"))
    end
    return collect(scales)
end

"""
    _scaled_inputs_and_norms(X, scale_values)

Scale the columns of `X` by `scale_values` and return both the scaled data and
their row-wise squared norms. Handles mixed element types via promotion.
"""
function _scaled_inputs_and_norms(X::AbstractMatrix, scale_values::AbstractVector)
    scaled_inputs = X .* reshape(scale_values, 1, :)
    norms = vec(sum(abs2, scaled_inputs; dims = 2))
    return scaled_inputs, norms
end

end
