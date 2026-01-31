using KernelFunctions

struct MonomialWeightedKernel{TK, Order} <: KernelFunctions.Kernel
    kernel::TK
end

function MonomialWeightedKernel(k::KernelFunctions.Kernel, order::Int)
    return MonomialWeightedKernel{typeof(k), order}(k)
end

function (k::MonomialWeightedKernel{TK, Order})(x, y) where {TK, Order}
    return x^Order * y * Order * k.kernel(x, y)
end
