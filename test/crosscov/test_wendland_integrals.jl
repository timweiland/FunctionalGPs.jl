using GaussPDE
using QuadGK
using Random


@testset "Wendland Integrals" begin
    ℓ = 7 // 10
    domains = intervals_from_endpoints(range(0, 4; step = 0.3))
    domains2 = intervals_from_endpoints(range(1, 5; step = 0.25))
    X = range(0, 3; step = 0.25)

    @testset "One-sided, k = $k, integral_arg = $integral_arg" for k in 1:3,
            integral_arg in [1, 2]

        w = WendlandKernel(1, k, ℓ)
        pv = IntegralPVCrosscov(w, domains, integral_arg)
        ℒ = EvaluationFunctional(X)
        K = Matrix(ℒ(pv))

        if integral_arg == 1
            # K[i,j] = cov(integrate over domains[i], evaluate at X[j])
            @test size(K) == (length(domains), length(X))
            for i in eachindex(domains), j in eachindex(X)
                @test K[i, j] ≈ covfunc_integral_one_sided_quad(
                    w,
                    X[j],
                    domains[i].lower,
                    domains[i].upper,
                ) rtol = 1.0e-3 atol = 1.0e-8
            end
        else  # integral_arg == 2
            # K[i,j] = cov(evaluate at X[i], integrate over domains[j])
            @test size(K) == (length(X), length(domains))
            for i in eachindex(X), j in eachindex(domains)
                @test K[i, j] ≈ covfunc_integral_one_sided_quad(
                    w,
                    X[i],
                    domains[j].lower,
                    domains[j].upper,
                ) rtol = 1.0e-3 atol = 1.0e-8
            end
        end
    end

    @testset "Two-sided, k = $k" for k in 1:3
        w = WendlandKernel(1, k, ℓ)
        K = kernel_integrate_integrate(w, domains, domains2)
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
            ) rtol = 1.0e-3 atol = 1.0e-8
        end
    end
end
