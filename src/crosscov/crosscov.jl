export ProcessVectorCrossCovariance, randvar_batch_size, randvar_length, randvar_arg, randproc_arg

abstract type ProcessVectorCrossCovariance end

function randvar_batch_size(pv_crosscov::ProcessVectorCrossCovariance)
    return throw(MethodError(randvar_batch_size, (pv_crosscov,)))
end
randvar_length(pv_crosscov::ProcessVectorCrossCovariance) = prod(randvar_batch_size(pv_crosscov))
function randvar_arg(pv_crosscov::ProcessVectorCrossCovariance)
    return throw(MethodError(randvar_arg, (pv_crosscov,)))
end
randproc_arg(pv_crosscov::ProcessVectorCrossCovariance) = (randvar_arg(pv_crosscov) == 1) ? 2 : 1

function kernelmatrix(pv::ProcessVectorCrossCovariance, X::AbstractVector)
    return throw(MethodError(kernelmatrix, (pv, X)))
end

# Concrete crosscov types
include("evaluation.jl")
include("integral.jl")
include("stacked.jl")

# Arithmetic operations on crosscovs
include("tensor_product.jl")
include("scale.jl")
include("sum.jl")
include("difference.jl")
