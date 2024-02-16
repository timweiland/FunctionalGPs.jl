export SumLinearFunctionOperator

abstract type SumLinearFunctionOperator <: AbstractLinearFunctionOperator end
summands(op::SumLinearFunctionOperator) = op.summands
function (op::SumLinearFunctionOperator)(args...)
    return sum([summand(args...) for summand in summands(op)])
end

function Base.show(io::IO, op::SumLinearFunctionOperator)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end