export StackedLinearFunctional

"""
    StackedLinearFunctional{N} <: AbstractLinearFunctional

A linear functional that stacks multiple linear functionals together.

When applied to a kernel from both sides (i.e., `ℒ1(ℒ2(kernel; arg=1); arg=2)` where both
`ℒ1` and `ℒ2` are `StackedLinearFunctional`s), produces a symmetric block matrix where:
- Diagonal blocks correspond to applying the same functional from both sides
- Off-diagonal blocks correspond to cross-covariances between different functionals

# Example
```julia
# Create a stacked functional with evaluation and integration
X = [1.0, 2.0, 3.0]
δ = EvaluationFunctional(X)
domains = [Interval(0.0, 1.0), Interval(1.0, 2.0)]
ℒ = VectorizedLebesgueIntegral(domains)

stacked = StackedLinearFunctional([δ, ℒ])

# Apply to kernel from both sides to get block matrix
k = SqExponentialKernel()
pv1 = stacked(k; arg=1)  # Apply to first argument
result = stacked(pv1)     # Apply to second argument, produces block matrix
```

The resulting block matrix has the structure:
```
┌─────────────┬──────────────┐
│  δ(k)(δ)    │  δ(k)(ℒ)     │
├─────────────┼──────────────┤
│  ℒ(k)(δ)    │  ℒ(k)(ℒ)     │
└─────────────┴──────────────┘
```

# Fields
- `functionals::Vector{AbstractLinearFunctional}`: The linear functionals to stack
"""
struct StackedLinearFunctional <: AbstractLinearFunctional
    functionals::Vector{AbstractLinearFunctional}

    function StackedLinearFunctional(functionals::Vector{<:AbstractLinearFunctional})
        if length(functionals) == 0
            throw(ArgumentError("At least one functional must be provided"))
        end
        return new(convert(Vector{AbstractLinearFunctional}, functionals))
    end
end

"""
    StackedLinearFunctional(functionals::AbstractLinearFunctional...)

Convenience constructor that accepts functionals as separate arguments.

# Example
```julia
δ = EvaluationFunctional([1.0, 2.0])
ℒ = VectorizedLebesgueIntegral([Interval(0.0, 1.0)])
stacked = StackedLinearFunctional(δ, ℒ)
```
"""
function StackedLinearFunctional(functionals::AbstractLinearFunctional...)
    return StackedLinearFunctional(collect(functionals))
end

"""
    output_shape(stacked::StackedLinearFunctional)

Returns a tuple of output shapes for each functional in the stack.
"""
function output_shape(stacked::StackedLinearFunctional)
    return tuple([output_shape(f) for f in stacked.functionals]...)
end

# Iterator interface for convenience
function Base.iterate(stacked::StackedLinearFunctional, state = 1)
    return state > length(stacked.functionals) ? nothing :
        (stacked.functionals[state], state + 1)
end

Base.length(stacked::StackedLinearFunctional) = length(stacked.functionals)
Base.getindex(stacked::StackedLinearFunctional, i) = stacked.functionals[i]

function Base.show(io::IO, stacked::StackedLinearFunctional)
    print(io, "StackedLinearFunctional([")
    join(io, stacked.functionals, ", ")
    print(io, "])")
end
