using ReTest
using GaussPDE
using Polynomials
import Random: seed!
import FiniteDifferences: central_fdm


const wendlands_d1 = [Polynomial([1]), Polynomial([1, 3]), Polynomial([1, 5, 8])]
# Note: Wendland(3, 2) is unnormalized in Table 9.1 of "Scattered Data Approximation"
const wendlands_d3 = [
    Polynomial([1]),
    Polynomial([1, 4]),
    Polynomial([1, 6, 35 // 3]),
    Polynomial([1, 8, 25, 32]),
]
const wendlands_d5 = [Polynomial([1]), Polynomial([1, 5]), Polynomial([1, 7, 16])]
const all_wendlands = [wendlands_d1, wendlands_d3, wendlands_d5]
const FD_ORDER = 16

@testset "Wendland Kernel" begin
    seed!(340954390)

    @testset "ϕ_l polynomial expansion (l=$l)" for l in 1:11
        for x in rand(10)
            @test isapprox(ϕ_l(l)(x), (1 - x)^l, atol = 1.0e-8)
        end
    end

    @testset "Wendland Polynomials (d=$(1 + 2 * d_iter))" for d_iter in 0:2
        d = 1 + 2 * d_iter
        wendland_polys_d = all_wendlands[d_iter + 1]
        @testset "k = $(k_iter - 1)" for k_iter in eachindex(wendland_polys_d)
            k = k_iter - 1
            cur_poly(r) = (1 - r)^(d_iter + 2 * k + 1) * wendland_polys_d[k_iter](r)
            for x in rand(10)
                @test cur_poly(x) ≈ WendlandPolynomial(d, k)(x) atol = 1.0e-8
            end
        end
    end

    @testset "Wendland Kernels: Compact Support" begin
        for d in 1:2:5
            for k in 0:2
                for ℓ in rand(1:10)
                    k = WendlandKernel(d, k, Float64(ℓ))
                    ε = rand(0.1:0.1:1.0)
                    @test k(0.0, ℓ) ≈ 0.0 atol = 1.0e-8
                    @test k(0.0, ℓ + ε) == 0.0
                end
            end
        end
    end

    function 𝒟k𝒟′_fd(k, n, m)
        return (x, y) ->
        central_fdm(FD_ORDER, m)(y -> central_fdm(FD_ORDER, n)(x -> k(x, y), x), y)
    end

    @testset "Wendland Derivatives" begin
        ℓ = 7 // 10
        @testset "d = $d, k=$k" for (d, k) in Iterators.product(1:2:5, 1:3)
            w = WendlandKernel(d, k, ℓ)
            @testset "n = $n, m=$m" for (n, m) in Iterators.product(0:k, 0:k)
                𝒟k𝒟′ = GaussPDE.derivative(w, n, m)
                if (n != 0) || (m != 0)
                    @test 𝒟k𝒟′ isa DerivativeKernel1D{n, m}
                end
                𝒟k𝒟′_approx = 𝒟k𝒟′_fd(w, n, m)
                lower = rand(0:0.3)
                upper = rand(0.3:0.7)
                out_of_support = 1.0 + rand()

                @test 𝒟k𝒟′(lower, upper) ≈ 𝒟k𝒟′_approx(lower, upper) rtol = 1.0e-3 atol =
                    1.0e-8
                @test 𝒟k𝒟′(upper, lower) ≈ 𝒟k𝒟′_approx(upper, lower) rtol = 1.0e-3 atol =
                    1.0e-8
                @test 𝒟k𝒟′(lower, lower) ≈ 𝒟k𝒟′_approx(lower, lower) rtol = 2.0e-1 atol =
                    1.0e-6
                @test 𝒟k𝒟′(lower, out_of_support) ≈ 0 atol = 1.0e-12
            end
        end
    end

    @testset "Consistency of concatenated derivatives" begin
        ℓ = 7 // 10

        @testset "d = $d, k=$k" for (d, k) in Iterators.product(1:2:5, 2:3)
            w = WendlandKernel(d, k, ℓ)
            D2k = GaussPDE.derivative(w, 2, 0)
            D2k_concat = GaussPDE.derivative(GaussPDE.derivative(w, 1, 0), 1, 0)
            @test D2k == D2k_concat

            Dk2 = GaussPDE.derivative(w, 0, 2)
            Dk2_concat = GaussPDE.derivative(GaussPDE.derivative(w, 0, 1), 0, 1)
            @test Dk2 == Dk2_concat

            DkD = GaussPDE.derivative(w, 1, 1)
            DkD_concat = GaussPDE.derivative(GaussPDE.derivative(w, 1, 0), 0, 1)
            @test DkD == DkD_concat

            if k == 3
                D3 = GaussPDE.derivative(w, 3, 0)
                D3_concat = GaussPDE.derivative(GaussPDE.derivative(w, 2, 0), 1, 0)
                @test D3 == D3_concat

                D2kD = GaussPDE.derivative(w, 2, 1)
                D2kD_concat = GaussPDE.derivative(GaussPDE.derivative(w, 1, 1), 1, 0)
                @test D2kD == D2kD_concat

                Dk2D = GaussPDE.derivative(w, 1, 2)
                Dk2D_concat = GaussPDE.derivative(GaussPDE.derivative(w, 1, 1), 0, 1)
                @test Dk2D == Dk2D_concat
            end
        end
    end

    @testset "Radial antiderivatives" begin
        ℓ = 0.6
        w = WendlandKernel(1, 3, ℓ)
        anti = radial_antiderivative(w, Val(1))
        rs = rand(10)
        for r in rs
            @test central_fdm(12, 1)(anti, r / ℓ) ≈ w(0.0, r) rtol = 1.0e-3 atol = 1.0e-8
        end

        anti2 = radial_antiderivative(w, Val(2))
        for r in rs
            @test central_fdm(12, 1)(anti2, r / ℓ) ≈ anti(r / ℓ) rtol = 1.0e-3 atol = 1.0e-8
        end
    end
end
