using FiniteDifferences: central_fdm
using QuadGK: quadgk
using KernelFunctions: SqExponentialKernel, ScaleTransform, ARDTransform
using Kronecker: kronecker

# ============================================================================
# Radial antiderivative tests
# ============================================================================

@testset "SE radial antiderivatives" begin
    @testset "Unit scale" begin
        k = SqExponentialKernel()

        @testset "First antiderivative" begin
            anti = radial_antiderivative(k, Val(1))
            # Derivative of antiderivative should equal kernel
            for r in [0.1, 0.5, 1.0, 2.0, 3.0]
                @test central_fdm(12, 1)(anti, r) ≈ k([0.0], [r]) rtol = 1.0e-3 atol = 1.0e-8
            end
        end

        @testset "Second antiderivative" begin
            anti1 = radial_antiderivative(k, Val(1))
            anti2 = radial_antiderivative(k, Val(2))
            # Derivative of second antiderivative should equal first
            for r in [0.1, 0.5, 1.0, 2.0, 3.0]
                @test central_fdm(12, 1)(anti2, r) ≈ anti1(r) rtol = 1.0e-3 atol = 1.0e-8
            end
        end
    end

    @testset "With ScaleTransform" begin
        ℓ = 0.6
        k = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)

        @testset "First antiderivative" begin
            anti = radial_antiderivative(k, Val(1))
            # Antiderivative is in terms of normalized distance τ = r/ℓ
            for r in [0.1, 0.3, 0.6, 1.0, 1.5]
                τ = r / ℓ
                @test central_fdm(12, 1)(anti, τ) ≈ k([0.0], [r]) rtol = 1.0e-3 atol = 1.0e-8
            end
        end

        @testset "Second antiderivative" begin
            anti1 = radial_antiderivative(k, Val(1))
            anti2 = radial_antiderivative(k, Val(2))
            for r in [0.1, 0.3, 0.6, 1.0, 1.5]
                τ = r / ℓ
                @test central_fdm(12, 1)(anti2, τ) ≈ anti1(τ) rtol = 1.0e-3 atol = 1.0e-8
            end
        end
    end
end

# ============================================================================
# Integral-evaluate cross-covariance tests
# ============================================================================

@testset "SE integral-evaluate cross-covariance" begin
    @testset "Unit scale" begin
        k = SqExponentialKernel()
        domains = [Interval(0.0, 0.4), Interval(0.5, 1.0), Interval(-0.2, 0.1)]
        X = collect(range(-0.3, 1.2; length = 5))

        lazy = kernel_integrate_evaluate(k, domains, X)

        # Validate against QuadGK
        for (i, dom) in enumerate(domains)
            for (j, x) in enumerate(X)
                expected, _ = quadgk(t -> k([t], [x]), dom.lower, dom.upper)
                @test lazy[i, j] ≈ expected rtol = 1.0e-6 atol = 1.0e-10
            end
        end
    end

    @testset "With ScaleTransform" begin
        ℓ = 1.3
        k = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)
        domains = [Interval(0.0, 0.5), Interval(0.6, 1.2)]
        X = collect(range(-0.1, 1.5; length = 4))

        lazy = kernel_integrate_evaluate(k, domains, X)

        for (i, dom) in enumerate(domains)
            for (j, x) in enumerate(X)
                expected, _ = quadgk(t -> k([t], [x]), dom.lower, dom.upper)
                @test lazy[i, j] ≈ expected rtol = 1.0e-6 atol = 1.0e-10
            end
        end
    end
end

# ============================================================================
# Integral-integral covariance tests
# ============================================================================

@testset "SE integral-integral covariance" begin
    @testset "Unit scale" begin
        k = SqExponentialKernel()
        domains_left = [Interval(-0.1, 0.3), Interval(0.4, 0.8)]
        domains_right = [Interval(0.0, 0.4), Interval(0.5, 1.0), Interval(0.2, 0.6)]

        lazy = kernel_integrate_integrate(k, domains_left, domains_right)

        # Validate against double QuadGK
        for (i, dom1) in enumerate(domains_left)
            for (j, dom2) in enumerate(domains_right)
                expected, _ = quadgk(
                    s -> quadgk(t -> k([s], [t]), dom2.lower, dom2.upper)[1],
                    dom1.lower, dom1.upper,
                )
                @test lazy[i, j] ≈ expected rtol = 1.0e-5 atol = 1.0e-9
            end
        end
    end

    @testset "With ScaleTransform" begin
        ℓ = 0.8
        k = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)
        domains_left = [Interval(0.0, 0.3), Interval(0.4, 0.7)]
        domains_right = [Interval(0.1, 0.5), Interval(0.6, 0.9)]

        lazy = kernel_integrate_integrate(k, domains_left, domains_right)

        for (i, dom1) in enumerate(domains_left)
            for (j, dom2) in enumerate(domains_right)
                expected, _ = quadgk(
                    s -> quadgk(t -> k([s], [t]), dom2.lower, dom2.upper)[1],
                    dom1.lower, dom1.upper,
                )
                @test lazy[i, j] ≈ expected rtol = 1.0e-5 atol = 1.0e-9
            end
        end
    end

    @testset "Symmetric case" begin
        k = SqExponentialKernel() ∘ ScaleTransform(1.2)
        domains = [Interval(0.0, 0.3), Interval(0.3, 0.6), Interval(0.6, 1.0)]

        lazy = kernel_integrate_integrate(k, domains)

        # Should be symmetric
        @test lazy ≈ lazy' atol = 1.0e-10

        # Validate diagonal entries
        for (i, dom) in enumerate(domains)
            expected, _ = quadgk(
                s -> quadgk(t -> k([s], [t]), dom.lower, dom.upper)[1],
                dom.lower, dom.upper,
            )
            @test lazy[i, i] ≈ expected rtol = 1.0e-5 atol = 1.0e-9
        end
    end
end

# ============================================================================
# Multi-dimensional (box domain) integral tests
# ============================================================================

@testset "SE multi-D box integrals" begin
    @testset "2D unit scale" begin
        k = SqExponentialKernel()

        # Create factorized box domains
        intervals1 = [Interval(0.0, 0.3), Interval(0.4, 0.7)]
        intervals2 = [Interval(0.1, 0.4), Interval(0.5, 0.8), Interval(0.2, 0.5)]
        box_domains = intervals1 ⊗ intervals2

        ∫_box = VectorizedLebesgueIntegral(box_domains)

        @testset "One-sided (integral-evaluate) structure" begin
            k∫ = ∫_box(k)
            @test k∫ isa TensorProductCrosscov{2}

            # Should decompose into 1D integrals
            ∫₁ = VectorizedLebesgueIntegral(intervals1)
            ∫₂ = VectorizedLebesgueIntegral(intervals2)
            k1 = SqExponentialKernel()
            k2 = SqExponentialKernel()

            @test k∫.factors[1] == ∫₁(k1)
            @test k∫.factors[2] == ∫₂(k2)
        end

        @testset "Two-sided (integral-integral) structure" begin
            k∫ = ∫_box(k)
            ∫k∫ = ∫_box(k∫)

            # Should be Kronecker product of 1D integral-integral matrices
            ∫₁ = VectorizedLebesgueIntegral(intervals1)
            ∫₂ = VectorizedLebesgueIntegral(intervals2)
            k1 = SqExponentialKernel()
            k2 = SqExponentialKernel()

            ∫₁k1∫₁ = ∫₁(∫₁(k1))
            ∫₂k2∫₂ = ∫₂(∫₂(k2))

            @test ∫k∫ ≈ kronecker(∫₂k2∫₂, ∫₁k1∫₁)
        end
    end

    @testset "2D with ARDTransform" begin
        ℓ1, ℓ2 = 0.5, 0.8
        k = SqExponentialKernel() ∘ ARDTransform([1 / ℓ1, 1 / ℓ2])

        intervals1 = [Interval(0.0, 0.2), Interval(0.3, 0.5)]
        intervals2 = [Interval(0.1, 0.3), Interval(0.4, 0.6)]
        box_domains = intervals1 ⊗ intervals2

        ∫_box = VectorizedLebesgueIntegral(box_domains)

        k∫ = ∫_box(k)
        @test k∫ isa TensorProductCrosscov{2}

        # Verify Kronecker structure
        ∫k∫ = ∫_box(k∫)

        ∫₁ = VectorizedLebesgueIntegral(intervals1)
        ∫₂ = VectorizedLebesgueIntegral(intervals2)
        k1 = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ1)
        k2 = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ2)

        ∫₁k1∫₁ = ∫₁(∫₁(k1))
        ∫₂k2∫₂ = ∫₂(∫₂(k2))

        @test ∫k∫ ≈ kronecker(∫₂k2∫₂, ∫₁k1∫₁)
    end

    @testset "2D with ScaleTransform (isotropic)" begin
        ℓ = 0.6
        k = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)

        intervals1 = [Interval(0.0, 0.25), Interval(0.3, 0.55)]
        intervals2 = [Interval(0.1, 0.35), Interval(0.4, 0.65)]
        box_domains = intervals1 ⊗ intervals2

        ∫_box = VectorizedLebesgueIntegral(box_domains)

        k∫ = ∫_box(k)
        @test k∫ isa TensorProductCrosscov{2}

        # Verify Kronecker structure
        ∫k∫ = ∫_box(k∫)

        ∫₁ = VectorizedLebesgueIntegral(intervals1)
        ∫₂ = VectorizedLebesgueIntegral(intervals2)
        k1 = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)
        k2 = SqExponentialKernel() ∘ ScaleTransform(1 / ℓ)

        ∫₁k1∫₁ = ∫₁(∫₁(k1))
        ∫₂k2∫₂ = ∫₂(∫₂(k2))

        @test ∫k∫ ≈ kronecker(∫₂k2∫₂, ∫₁k1∫₁)
    end
end
