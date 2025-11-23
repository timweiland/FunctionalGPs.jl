using ForwardDiff
using ToeplitzMatrices: SymmetricToeplitz
using Test

function _autodiff_matern_derivative(k, x, y, n, m)
    if n == 0 && m == 0
        return k(x, y)
    elseif n > 0
        return ForwardDiff.derivative(t -> _autodiff_matern_derivative(k, t, y, n - 1, m), x)
    else
        return ForwardDiff.derivative(t -> _autodiff_matern_derivative(k, x, t, n, m - 1), y)
    end
end

@testset "HalfIntegerMatern derivatives" begin
    k = HalfIntegerMaternKernel(2, [0.8])
    x = 0.3
    y = 1.1
    orders = [(1, 0), (0, 1), (2, 0), (1, 1), (3, 0)]

    for (n, m) in orders
        D = GaussPDE.derivative(k, n, m)
        expected = _autodiff_matern_derivative(k, x, y, n, m)
        @test D(x, y) ≈ expected atol = 1.0e-9 rtol = 1.0e-7
    end

    D_odd = GaussPDE.derivative(k, 1, 0)
    @test D_odd(0.5, 0.5) == 0

    D_even = GaussPDE.derivative(k, 2, 0)
    @test kernel_structure(D_even.derivative_kernel) isa StationaryKernelTrait
    grid = range(0.0, stop = 1.0, length = 6)
    mat = kernel_evaluate_evaluate(D_even.derivative_kernel, grid)
    @test mat isa SymmetricToeplitz

    X_left = collect(range(0.0, stop = 0.8, length = 5))
    X_right = collect(range(0.1, stop = 0.9, length = 5))
    lazy_cross = kernel_evaluate_evaluate(k, X_left, X_right)
    dense_cross = kernelmatrix(k, X_left, X_right)
    @test lazy_cross ≈ dense_cross atol = 1.0e-9 rtol = 1.0e-7
end
