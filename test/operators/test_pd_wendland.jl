@testset "Partial Derivative of Wendland kernels" begin
    @testset "d = $d, k=$k, l=$ℓ" for (d, k, ℓ) in Iterators.product(1:2:5, 1:3, [7 // 10, 1])
        w = ℓ == 1 ? WendlandKernel(d, k) : WendlandKernel(d, k, ℓ)
        @testset "n = $n, m=$m" for (n, m) in Iterators.product(0:k, 0:k)
            𝒟₁ = PartialDerivative((n,))
            𝒟₂ = PartialDerivative((m,))
            @test 𝒟₂(w) ≈ FunctionalGPs.derivative(w, 0, m)
            @test 𝒟₁(w, arg = 1) ≈ FunctionalGPs.derivative(w, n, 0)
            @test 𝒟₁(𝒟₂(w), arg = 1) ≈ FunctionalGPs.derivative(w, n, m)
            @test_throws DomainError 𝒟₂(w, arg = 3)
        end
    end
end
