# TensorProductFunctional applied to PVCrosscovs → creates matrices

using Kronecker

"""
Application of TensorProductFunctional to TensorProductCrosscov.

Produces the Kronecker product of individual applications.
The result is ordered with the last dimension varying fastest (column-major).
"""
function (ℒ::TensorProductFunctional{N})(pv::TensorProductCrosscov{N}) where {N}
    return mapreduce(
        args -> ((cur_pv, cur_ℒ) = args; cur_ℒ(cur_pv)),
        kronecker,
        zip(pv.factors, ℒ.factors) |> collect |> reverse,
    )
end
