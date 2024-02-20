export ProcessVectorCrossCovariance, randvar_batch_size, randvar_length, randvar_arg, randproc_arg

abstract type ProcessVectorCrossCovariance end

function randvar_batch_size(pv_crosscov::ProcessVectorCrossCovariance)
    return error("randvar_batch_size not implemented for $(typeof(pv_crosscov))")
end
randvar_length(pv_crosscov::ProcessVectorCrossCovariance) = prod(randvar_batch_size(pv_crosscov))
function randvar_arg(pv_crosscov::ProcessVectorCrossCovariance)
    return error("randvar_arg not implemented for $(typeof(pv_crosscov))")
end
randproc_arg(pv_crosscov::ProcessVectorCrossCovariance) = (randvar_arg(pv_crosscov) == 1) ? 2 : 1

function kernelmatrix(pv::ProcessVectorCrossCovariance, ::AbstractVector)
    return error("kernelmatrix not implemented for $(typeof(pv))")
end
