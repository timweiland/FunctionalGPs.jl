struct OverlapKernel1D{TK, TD, TDL, TFL, TDR, TFR} <: KernelFunctions.Kernel
    kernel::TK
    inner_domain::TD
    left_overlap_domain::TDL
    weight_fn_left::TFL
    right_overlap_domain::TDR
    weight_fn_right::TFR
end

function _get_weight(k::OverlapKernel1D, x)
    if x in k.inner_domain
        return 1.0
    elseif (k.left_overlap_domain !== nothing) && (x in k.left_overlap_domain)
        return k.weight_fn_left(x)
    elseif (k.right_overlap_domain !== nothing) && (x in k.right_overlap_domain)
        return k.weight_fn_right(x)
    else
        return 0.0
    end
end

function (k::OverlapKernel1D)(x, y)
    return _get_weight(k, x) * _get_weight(k, y) * k.kernel(x, y)
end

function KernelFunctions.kernelmatrix(k::OverlapKernel1D, X::AbstractVector, Y::AbstractVector)
    wX = map(x -> _get_weight(k, x), X)
    wY = map(y -> _get_weight(k, y), Y)
    K0 = kernelmatrix(k.kernel, X, Y)
    @inbounds for j in eachindex(Y), i in eachindex(X)
        K0[i, j] *= wX[i] * wY[j]
    end
    return K0
end
