using GaussPDE
using QuadGK
using Random

covfunc_integral_one_sided_quad(k, x, a, b) = quadgk(y -> k(x, y), a, b)[1]
function covfunc_integral_two_sided_quad(k, a, b, c, d)
    return quadgk(x -> covfunc_integral_one_sided_quad(k, x, c, d), a, b)[1]
end

@testset "Wendland Integrals" begin
    ℓ = 7 // 10
    domains = intervals_from_endpoints(range(0, 4; step = 0.3))
    domains2 = intervals_from_endpoints(range(1, 5; step = 0.25))
    X = range(0, 3; step = 0.25)

    @testset "One-sided, k = $k, randvar_arg = $randvar_arg" for k in 1:3,
        randvar_arg in [1, 2]

        w = WendlandKernel(1, k, ℓ)
        pv = CompactPolynomialCovFunc1D_Identity_LebesgueIntegral(w, domains, randvar_arg)
        K = kernelmatrix(pv, X)
        if randvar_arg == 2
            K = K'
        end
        @test size(K) == (length(domains), length(X))
        for i in eachindex(domains), j in eachindex(X)
            @test K[i, j] ≈ covfunc_integral_one_sided_quad(
                w,
                X[j],
                domains[i].lower,
                domains[i].upper,
            ) rtol = 1e-3 atol = 1e-8
        end
    end

    @testset "Two-sided, k = $k" for k in 1:3
        w = WendlandKernel(1, k, ℓ)
        K = GaussPDE.integrate(w, domains, domains2)
        @test size(K) == (length(domains), length(domains2))
        all_idcs = collect(Iterators.product(eachindex(domains), eachindex(domains2)))
        rand_idcs = all_idcs[randperm(length(all_idcs))[1:10]]
        for (i, j) in rand_idcs
            @test K[i, j] ≈ covfunc_integral_two_sided_quad(
                w,
                domains[i].lower,
                domains[i].upper,
                domains2[j].lower,
                domains2[j].upper,
            ) rtol = 1e-3 atol = 1e-8
        end
    end
end