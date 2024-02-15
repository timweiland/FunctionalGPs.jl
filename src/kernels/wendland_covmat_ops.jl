import AbstractGPs: Xt_A_X, Xt_A_Y, Xt_invA_X, Xt_invA_Y, diag_Xt_A_X, diag_Xt_A_Y, diag_Xt_invA_X, diag_Xt_invA_Y, tr_Xt_invA_X, Xtinv_A_Xinv, diag_At_A
import SparseArrays.CHOLMOD.Factor as SparseCholFactor

export Xt_A_X, Xt_A_Y, Xt_invA_X, Xt_invA_Y, diag_Xt_A_X, diag_Xt_A_Y, diag_Xt_invA_X, diag_Xt_invA_Y, tr_Xt_invA_X, Xtinv_A_Xinv

Xt_A_X(A::SparseCholFactor, x::AbstractVector) = sum(abs2, A.UP * x)
Xt_A_X(A::SparseCholFactor, X::AbstractMatrix) = (V = A.UP * X; Symmetric(V'V))

Xt_A_Y(X::AbstractVecOrMat, A::SparseCholFactor, Y::AbstractVecOrMat) = (A.UP * X)' * (A.UP * Y)

Xt_invA_X(A::SparseCholFactor, x::AbstractVector) = sum(abs2, A.PtL \ x)
Xt_invA_X(A::SparseCholFactor, X::AbstractMatrix) = (V = A.PtL \ X; Symmetric(V'V))

Xt_invA_Y(X::AbstractVecOrMat, A::SparseCholFactor, Y::AbstractVecOrMat) = (A.PtL\ X)' * (A.PtL \ Y)

diag_Xt_A_X(A::SparseCholFactor, X::AbstractVecOrMat) = diag_At_A(A.UP * X)
diag_Xt_A_Y(X::AbstractVecOrMat, A::SparseCholFactor, Y::AbstractVecOrMat) = diag_At_B(A.UP * X, A.UP * Y)

diag_Xt_invA_X(A::SparseCholFactor, X::AbstractVecOrMat) = diag_At_A(A.PtL \ X)
diag_Xt_invA_Y(X::AbstractVecOrMat, A::SparseCholFactor, Y::AbstractMatrix) = diag_At_B(A.PtL \ X, A.PtL \ Y)

tr_Xt_invA_X(A::SparseCholFactor, X::AbstractVecOrMat) = tr_At_A(A.PtL \ X)
Xtinv_A_Xinv(A::SparseCholFactor, X::Cholesky) = (C = A.UP \ (X.L \ A.PtL); Symmetric(C*C'))
Xtinv_A_Xinv(A::SparseCholFactor, X::SparseCholFactor) = (C = A.UP \ (X.PtL \ A.PtL); Symmetric(C*C'))
