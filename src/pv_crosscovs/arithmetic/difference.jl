function Base.:(-)(pv1::ProcessVectorCrossCovariance, pv2::ProcessVectorCrossCovariance)
    return pv1 + (-1 * pv2)
end
