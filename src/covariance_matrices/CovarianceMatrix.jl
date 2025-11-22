export CovarianceMatrix

"""
    CovarianceMatrix{T}

Abstract supertype for covariance-like matrices with element type `T`.

Subtypes should implement the standard array interface (`size`, `axes`,
`getindex`, etc.) while delaying materialisation whenever possible. The type
hierarchy enables lazy covariance algebra without modifying generic
`AbstractMatrix` behaviour.
"""
abstract type CovarianceMatrix{T} <: AbstractMatrix{T} end
