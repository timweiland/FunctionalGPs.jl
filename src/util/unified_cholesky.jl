using LinearAlgebra: Cholesky
using SparseArrays: CHOLMOD
using LinearOperators: LinearOperator

export matrix_sqrt, UnifiedCholesky

const UnifiedCholesky = Union{Cholesky, CHOLMOD.Factor}

function matrix_sqrt(C::Cholesky)
    return LinearOperator(C.L)
end

function matrix_sqrt(C::CHOLMOD.Factor)
    Ls = sparse(C.L)
    Pinv = invperm(C.p)
    prod_fn!(y, x) = (mul!(y, Ls, x); permute!(y, Pinv))
    t_prod_fn!(y, x) = (xp = copy(x); permute!(xp, C.p); mul!(y, Ls', xp))
    return LinearOperator(Float64, size(C.L, 1), size(C.L, 2), false, false, prod_fn!, t_prod_fn!, t_prod_fn!)
end
