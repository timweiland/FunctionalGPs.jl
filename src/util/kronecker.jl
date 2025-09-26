using Kronecker

function Base.:\(
        K::CholeskyKronecker,
        rhs::AbstractVector,
    )
    A = K.A
    B = K.B
    nA = size(A, 1)
    nB = size(B, 1)

    length(rhs) == nA * nB ||
        throw(DimensionMismatch("length(rhs)=$(length(rhs)) but nA*nB=$(nA * nB)"))

    # Reshape vec(R) -> R ∈ ℝ^{nB×nA}
    R = reshape(rhs, nB, nA)

    # Solve B * Y = R
    Y = B \ R

    # Solve A' * X' = Y'  ⇒  X = (A' \ Y')'
    X = (transpose(A) \ transpose(Y))'

    return vec(X)
end

function Base.:\(
        K::CholeskyKronecker,
        rhs::AbstractMatrix,
    )
    A = K.A
    B = K.B
    nA = size(A, 1)
    nB = size(B, 1)

    s1, s2 = size(rhs)
    s1 == nA * nB ||
        throw(DimensionMismatch("size(rhs,1)=$s1 but nA*nB=$(nA * nB)"))

    out = similar(rhs)  # (nA*nB)×nrhs

    @inbounds for j in 1:s2
        # Treat each column independently
        Rj = reshape(view(rhs, :, j), nB, nA)

        # Yj solves B * Yj = Rj
        Yj = B \ Rj

        # Xj solves A' * Xj' = Yj'  ⇒  Xj = (A' \ Yj')'
        Xj = (transpose(A) \ transpose(Yj))'

        # Store vec(Xj) back to output column
        copyto!(view(out, :, j), vec(Xj))
    end

    return out
end
