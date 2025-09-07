using KernelFunctions: Kernel

export radial_antiderivative

radial_antiderivative(k::Kernel, n::Int) = throw(MethodError(radial_antiderivative, (k, n)))
