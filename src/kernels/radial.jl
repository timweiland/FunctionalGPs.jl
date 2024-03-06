using KernelFunctions: Kernel

export radial_antiderivative

radial_antiderivative(k::Kernel, ::Int) = error("radial_antiderivative not implemented for $(typeof(k))")
