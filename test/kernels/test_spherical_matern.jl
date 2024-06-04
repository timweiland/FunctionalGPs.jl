include("../../src/kernels/matern.jl")

using .MaternKernelModule

# Create an intrinsic MaternKernel instance
# just using a dummy value for the variance
intrinsic_matern_kernel = IntrinsicMatern(MaternKernel(1.5, 2.0, 1.0))

# Create an extrinsic MaternKernel instance
# just using a dummy value for the variance
extrinsic_matern_kernel = ExtrinsicMatern(MaternKernel(1.5, 2.0, 1.0))

# Define angles for evaluation
θ1, φ1 = π / 4, π / 4
θ2, φ2 = π / 3, π / 3

# Evaluate intrinsic Matern kernel
result_intrinsic = evaluate(intrinsic_matern_kernel,θ1, φ1, θ2, φ2)

# Evaluate extrinsic Matern kernel
result_extrinsic = evaluate(extrinsic_matern_kernel, θ1, φ1, θ2, φ2)

println("Intrinsic Matern kernel evaluation result: $result_intrinsic")
println("Extrinsic Matern kernel evaluation result: $result_extrinsic")


# TODO: Implement a more sophisticated test for both the intrinsic and extrinsic Matern kernel