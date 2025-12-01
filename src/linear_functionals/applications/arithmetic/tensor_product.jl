import KernelFunctions: KernelTensorProduct
using Kronecker

"""
Application of TensorProductFunctional to KernelTensorProduct.

For `ℒ = ℒ₁ ⊗ ℒ₂ ⊗ ... ⊗ ℒₙ` and `k = k₁ ⊗ k₂ ⊗ ... ⊗ kₙ`:
    ℒ(k) = ℒ₁(k₁) ⊗ ℒ₂(k₂) ⊗ ... ⊗ ℒₙ(kₙ)
"""
function (ℒ::TensorProductFunctional{N})(k::KernelTensorProduct; arg = 2) where {N}
    if length(k.kernels) != N
        throw(
            ArgumentError(
                "Number of functionals ($N) must match number of kernel factors ($(length(k.kernels)))",
            ),
        )
    end
    return mapreduce(
        args -> ((cur_k, cur_ℒ) = args; cur_ℒ(cur_k; arg = arg)),
        ⊗,
        zip(k.kernels, ℒ.factors),
    )
end

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
