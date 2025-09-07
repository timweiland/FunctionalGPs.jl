using GaussPDE
using Test

@testset "CompactKernel kernelmatrix scalar ℓ" begin
    # 1D Wendland with scalar lengthscale should work with kernelmatrix
    ℓ = 0.6
    k = WendlandKernel(1, 2, ℓ)
    x = collect(range(0.0, 2.0; length=25))
    y = collect(range(0.0, 2.0; length=20))

    Ks = kernelmatrix(k, x, y)
    Kd = [k(xi, yj) for xi in x, yj in y]
    @test size(Ks) == (length(x), length(y))
    @test Matrix(Ks) ≈ Kd atol = 1e-12
end

