export Matern1D_Identity_LebesgueIntegral

struct Matern1D_Identity_LebesgueIntegral{TC, TD} <: RadialCovarianceFunction1D_Identity_LebesgueIntegral
    covfunc::TC
    domains::TD
    randvar_arg::Int
end
