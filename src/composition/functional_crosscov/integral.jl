# VectorizedLebesgueIntegral applied to PVCrosscovs → creates matrices

using Kronecker

# Apply to EvaluationPVCrosscov
function (ℒ::VectorizedLebesgueIntegral)(pv::EvaluationPVCrosscov)
    # Create IntegralPVCrosscov by applying ℒ to the kernel, then evaluate at the points
    return kernelmatrix(ℒ(pv.k, arg = randproc_arg(pv)), pv.linfunc.X)
end

# Apply to IntegralPVCrosscov (same domains → symmetric case)
function (ℒ::VectorizedLebesgueIntegral{Interval{T}})(pv::IntegralPVCrosscov) where {T}
    if ℒ.domains === pv.domains
        return kernel_integrate_integrate(pv.k, pv.domains)
    end
    return kernel_integrate_integrate(pv.k, ℒ.domains, pv.domains)
end

# Apply to TensorProductCrosscov with box domains
function box_integrals(pv::TensorProductCrosscov, domains::FactorizedBoxDomains)
    ℒs = map(VectorizedLebesgueIntegral, get_intervals(domains))
    return mapreduce(
        args -> (
            (cur_pv, cur_ℒ) = args;
            cur_ℒ(cur_pv)
        ),
        kronecker,
        zip(pv.factors, ℒs) |> collect |> reverse,
    )
end

function (ℒ::VectorizedLebesgueIntegral{BoxDomain{T}})(pv::TensorProductCrosscov) where {T}
    return box_integrals(pv, ℒ.domains)
end

# PartialDerivative applied to IntegralPVCrosscov (special case)
function (op::PartialDerivative{1, 1})(pv::IntegralPVCrosscov)
    k = pv.k
    dk = op(k; arg = randproc_arg(pv))
    ℒ = VectorizedLebesgueIntegral(pv.domains)
    return ℒ(dk; arg = randvar_arg(pv))
end
