export ZeroPVCrosscov

"""
    ZeroPVCrosscov

PV crosscovariance representing the identically-zero cross-covariance that
arises when a linear functional is applied to the zero kernel. Stores only the
shape of the random-variable dimension and which kernel argument it came from;
all covariance matrices it produces are zero matrices of the appropriate size.
"""
struct ZeroPVCrosscov <: ProcessVectorCrossCovariance
    batch_size::Tuple{Vararg{Int}}
    randvar_arg::Int
end

randvar_batch_size(pv::ZeroPVCrosscov) = pv.batch_size
randvar_arg(pv::ZeroPVCrosscov) = pv.randvar_arg

function kernelmatrix(pv::ZeroPVCrosscov, X::AbstractVector)
    n = randvar_length(pv)
    m = length(X)
    return pv.randvar_arg == 1 ? zeros(n, m) : zeros(m, n)
end
