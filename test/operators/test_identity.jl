using FunctionalGPs
using AbstractGPs
using KernelFunctions
using KernelFunctions: with_lengthscale, SqExponentialKernel, ScaledKernel, KernelSum
using LinearAlgebra

@testset "Identity operator" begin
    k_w = WendlandKernel(1, 3, 8 // 10)
    f = GP(k_w)
    X = collect(0.0:0.1:1.0)
    Y = collect(0.05:0.1:0.95)
    𝒟 = PartialDerivative((1,))
    I_op = Identity()

    @testset "Construction & show" begin
        @test I_op isa AbstractLinearFunctionOperator
        @test !(I_op isa AbstractDifferentialOperator)
        @test Identity() === I_op   # singleton
        @test string(I_op) == "I"
    end

    @testset "Apply to kernels — pass-through" begin
        # Bare Kernel — argument returned unchanged regardless of `arg`.
        @test I_op(k_w) === k_w
        @test I_op(k_w; arg = 1) === k_w
        @test I_op(k_w; arg = 2) === k_w

        # KernelSum
        ks = KernelSum((k_w, with_lengthscale(SqExponentialKernel(), 0.3)))
        @test I_op(ks) === ks
        @test I_op(ks; arg = 1) === ks

        # ScaledKernel (positive scale → ScaledKernel path)
        sk = ScaledKernel(k_w, 2.5)
        @test I_op(sk) === sk

        # LinearlyScaledKernel (negative scale path)
        lsk = LinearlyScaledKernel(k_w, -2.0)
        @test I_op(lsk) === lsk

        # SE kernels go through the tensor-product specialisation for derivatives;
        # Identity must NOT take that path — should just return the SE kernel.
        k_se = with_lengthscale(SqExponentialKernel(), 0.3)
        @test I_op(k_se) === k_se
    end

    @testset "Apply to mean — pass-through" begin
        @test I_op(f.mean) === f.mean
        # ZeroMean specifically (this is the type the abstract dispatch
        # already covers for other operators; verify Identity matches behaviour)
        zm = AbstractGPs.ZeroMean{Float64}()
        @test I_op(zm) isa AbstractGPs.ZeroMean{Float64}
    end

    @testset "Apply to PV crosscovs — pass-through" begin
        L_y = EvaluationFunctional(X)
        L_z = EvaluationFunctional(Y)
        pv_eval = L_y(f.kernel)            # EvaluationPVCrosscov
        @test I_op(pv_eval) === pv_eval

        pv_stack = StackedPVCrosscov([L_y(f.kernel), L_z(f.kernel)])
        @test I_op(pv_stack) === pv_stack

        pv_sum = 𝒟(f.kernel) + 𝒟(f.kernel)
        @test I_op(pv_sum) === pv_sum

        pv_scaled = 3.0 * L_y(f.kernel)
        @test I_op(pv_scaled) === pv_scaled
    end

    @testset "Composition with functional" begin
        L_y = EvaluationFunctional(X)
        ℒ_id = L_y ∘ I_op            # AbstractLinFctlLinFuncOpConcat
        @test ℒ_id isa AbstractLinFctlLinFuncOpConcat

        # Equivalent to plain L_y on a kernel
        @test ℒ_id(f.kernel) ≈ L_y(f.kernel)
        # And on the mean
        @test ℒ_id(f.mean) == L_y(f.mean)
        # And as a finite GP
        m_ref = L_y(f)
        m_id = ℒ_id(f)
        @test mean(m_id) ≈ mean(m_ref)
        @test cov(m_id) ≈ cov(m_ref)
    end

    @testset "Composition with another operator" begin
        # Identity ∘ ∂x == ∂x  (semantically)
        comp1 = I_op ∘ 𝒟
        @test comp1 isa AbstractConcatenatedLinearFunctionOperator
        @test comp1(f.kernel) ≈ 𝒟(f.kernel)

        # ∂x ∘ Identity == ∂x
        comp2 = 𝒟 ∘ I_op
        @test comp2(f.kernel) ≈ 𝒟(f.kernel)

        # Identity ∘ Identity
        comp3 = I_op ∘ I_op
        @test comp3(k_w) === k_w
    end

    @testset "Sum: Identity + ∂x acts as k + ∂k" begin
        op = I_op + 𝒟
        @test op isa AbstractSumLinearFunctionOperator{2}

        # The natural use is one-sided: op acts on a kernel; a functional
        # then reads it. Verify (I + ∂)(k) is k + ∂k by checking the kernel
        # matrix once the functional is applied.
        L_y = EvaluationFunctional(X)
        K_combined = Matrix(L_y(L_y(op(f.kernel))))
        K_ref = Matrix(L_y(L_y(f.kernel))) + Matrix(L_y(L_y(𝒟(f.kernel))))
        @test K_combined ≈ K_ref
    end

    @testset "Scale: α * Identity acts as α·k" begin
        op = 2.5 * I_op
        @test op isa ConstantScaledLinearFunctionOperator
        # Apply Identity-scaled to k → kernel with scaled values
        L_y = EvaluationFunctional(X)
        K_scaled = L_y(L_y(op(f.kernel)))
        @test Matrix(K_scaled) ≈ 2.5 .* Matrix(L_y(L_y(f.kernel)))

        # 1 * Identity short-circuits back to Identity (per Scale's special case)
        @test 1 * I_op === I_op
    end

    @testset "FunctionalGaussian with EvaluationFunctional ∘ Identity" begin
        L_y = EvaluationFunctional(X)
        fg_plain = FunctionalGaussian(f; y = L_y)
        fg_id = FunctionalGaussian(f; y = L_y ∘ I_op)
        @test mean(fg_id) ≈ mean(fg_plain)
        @test Matrix(cov(fg_id)) ≈ Matrix(cov(fg_plain))
    end
end
