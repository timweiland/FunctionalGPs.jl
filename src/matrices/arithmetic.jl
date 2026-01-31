import LazyArrays: ApplyArray, BroadcastArray

"""
    +(A::CovarianceMatrix, B::CovarianceMatrix)

Return a lazy representation of the elementwise sum using `ApplyArray`. Both
inputs must share the same shape.
"""
function Base.:+(A::CovarianceMatrix, B::CovarianceMatrix)
    size(A) == size(B) || throw(DimensionMismatch("covariance matrices must have matching sizes"))
    return ApplyArray(+, A, B)
end

"""
    -(A::CovarianceMatrix, B::CovarianceMatrix)

Lazy subtraction implemented via `ApplyArray`.
"""
function Base.:-(A::CovarianceMatrix{T}, B::CovarianceMatrix) where {T}
    if A === B
        return zero(T) * A
    end
    size(A) == size(B) || throw(DimensionMismatch("covariance matrices must have matching sizes"))
    return ApplyArray(+, A, B * (-1))
end

"""
    *(α::Number, A::CovarianceMatrix)
    *(A::CovarianceMatrix, α::Number)

Scale covariance matrices lazily using `BroadcastArray`.
"""
Base.:*(α::Number, A::CovarianceMatrix) = BroadcastArray(*, A, α)
Base.:*(A::CovarianceMatrix, α::Number) = BroadcastArray(*, A, α)

"""
    /(A::CovarianceMatrix, α::Number)

Divide covariance matrix entries by a scalar lazily.
"""
Base.:/(A::CovarianceMatrix, α::Number) = BroadcastArray(/, A, α)

"""
    +(A::CovarianceMatrix, α::Number)
    +(α::Number, A::CovarianceMatrix)

Add a scalar to every entry using `BroadcastArray`.
"""
Base.:+(A::CovarianceMatrix, α::Number) = BroadcastArray(+, A, α)
Base.:+(α::Number, A::CovarianceMatrix) = BroadcastArray(+, A, α)

"""
    -(A::CovarianceMatrix, α::Number)
    -(α::Number, A::CovarianceMatrix)

Subtract scalars from covariance matrices lazily.
"""
Base.:-(A::CovarianceMatrix, α::Number) = BroadcastArray(-, A, α)
Base.:-(α::Number, A::CovarianceMatrix) = BroadcastArray(-, α, A)
