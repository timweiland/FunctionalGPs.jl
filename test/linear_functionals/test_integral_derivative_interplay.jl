using GaussPDE
using QuadGK

covfunc_integral_one_sided_quad(k, x, a, b) = quadgk(y -> k(x, y), a, b)[1]
function covfunc_integral_two_sided_quad(k, a, b, c, d)
    return quadgk(x -> covfunc_integral_one_sided_quad(k, x, c, d), a, b)[1]
end

function random_idcs(K, n)
    all_idcs = collect(Iterators.product(eachindex(axes(K, 1)), eachindex(axes(K, 2))))
    return all_idcs[randperm(length(all_idcs))[1:n]]
end

@testset "Integral / Derivative interplay" begin
    k = WendlandKernel(1, 3, 0.6)
    dx = PartialDerivative((1,))
    intervals = intervals_from_endpoints(range(0, 3; step = 0.3))
    ∫ = VectorizedLebesgueIntegral(intervals)
    fv_op = ∫ ∘ dx

    @testset "Standard FV operator" begin
        @testset "One-sided" begin
            X = range(0, 3; step = 0.25)

            k_fv = fv_op(k)
            K = kernelmatrix(k_fv, X)
            @test size(K) == (length(X), length(intervals))
            idcs = random_idcs(K, 10)
            for (i, j) in idcs
                @test K[i, j] ≈ covfunc_integral_one_sided_quad(
                    dx(k),
                    X[i],
                    intervals[j].lower,
                    intervals[j].upper,
                ) rtol = 1e-3 atol = 1e-8
            end
        end

        @testset "Two-sided" begin
            K = fv_op(fv_op(k))
            @test size(K) == (length(intervals), length(intervals))
            idcs = random_idcs(K, 10)
            for (i, j) in idcs
                @test K[i, j] ≈ covfunc_integral_two_sided_quad(
                    dx(dx(k), arg=1),
                    intervals[i].lower,
                    intervals[i].upper,
                    intervals[j].lower,
                    intervals[j].upper,
                ) rtol = 1e-3 atol = 1e-8
            end
        end
    end

    @testset "Crossplay between arguments" begin
        dk = dx(k, arg=1)
        dk∫ = ∫(dk, arg=2)

        X = range(0, 3; step = 0.25)
        K = kernelmatrix(dk∫, X)
        @test size(K) == (length(X), length(intervals))
        idcs = random_idcs(K, 10)
        for (i, j) in idcs
            @test K[i, j] ≈ covfunc_integral_one_sided_quad(
                dk,
                X[i],
                intervals[j].lower,
                intervals[j].upper,
            ) rtol = 1e-3 atol = 1e-8
        end
    end
end
