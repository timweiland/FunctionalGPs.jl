export EvaluationFunctional

struct EvaluationFunctional <: AbstractLinearFunctional
    X::AbstractVector
    output_shape::Tuple{Vararg{Integer}}
end

function EvaluationFunctional(X::AbstractVector)
    return EvaluationFunctional(X, size(X))
end
