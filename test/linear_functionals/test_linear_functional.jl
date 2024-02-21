using GaussPDE
import GaussPDE: randvar_batch_size
using AbstractGPs
using KernelFunctions
import KernelFunctions: kernelmatrix

struct SillyFunctional <: AbstractLinearFunctional 
    output_shape::Tuple{Vararg{Int}}
end
struct SillyPVCrosscov <: ProcessVectorCrossCovariance 
    randvar_arg::Int
    randvar_shape::Tuple{Vararg{Int}}
end
randvar_batch_size(pv::SillyPVCrosscov) = pv.randvar_shape

(ℒ::SillyFunctional)(::Kernel; arg=2) = SillyPVCrosscov(arg, ℒ.output_shape)
function kernelmatrix(pv::SillyPVCrosscov, X::AbstractVector)
    if pv.randvar_arg == 1
        return 42 * ones(randvar_length(pv), length(X))
    else
        return 42 * ones(length(X), randvar_length(pv))
    end
end

@testset "Linear Functional default implementations" begin
    ℒ = SillyFunctional((3,))
    δ = EvaluationFunctional(rand(10))
    δ2 = EvaluationFunctional(rand(4))

    f = GP(WendlandKernel(1, 3))

    k_stacked = StackedPVCrosscov([δ(f.kernel), δ2(f.kernel)])
    stacked_k = StackedPVCrosscov([δ(f.kernel, arg=1), δ2(f.kernel, arg=1)])

    @test ℒ(f.mean) == zeros(3)
    @test ℒ(δ(f.kernel, arg=2)) ≈ 42 * ones(3, 10)
    @test ℒ(δ(f.kernel, arg=1)) ≈ 42 * ones(10, 3)
    ℒ_k_stacked = ℒ(k_stacked)
    @test ℒ_k_stacked ≈ hcat(42 * ones(3, 10), 42 * ones(3, 4))
    stacked_k_ℒ = ℒ(stacked_k)
    @test stacked_k_ℒ ≈ vcat(42 * ones(10, 3), 42 * ones(4, 3))
end
