export StackedLinearFunctional

"""
    StackedLinearFunctional <: AbstractLinearFunctional

A linear functional that stacks multiple linear functionals together. When applied to a kernel
from both sides, it creates a symmetric block matrix where each block represents the cross-covariance
between pairs of functionals.

# Fields
- `linfunctionals::Tuple`: A tuple of linear functionals to stack

# Example
```julia
δ1 = EvaluationFunctional(X1)
δ2 = EvaluationFunctional(X2)
stacked = StackedLinearFunctional(δ1, δ2)  # or StackedLinearFunctional([δ1, δ2])

# Apply to kernel to create StackedPVCrosscov
pv_stack = stacked(kernel)

# Apply again to create block matrix
block_matrix = stacked(pv_stack)  # Creates [[δ1∘δ1, δ1∘δ2], [δ2∘δ1, δ2∘δ2]]
```
"""
struct StackedLinearFunctional{Fs <: Tuple{Vararg{AbstractLinearFunctional}}} <: AbstractLinearFunctional
    linfunctionals::Fs
end

# Convenience constructors
StackedLinearFunctional(lfs::AbstractLinearFunctional...) = StackedLinearFunctional(lfs)

function StackedLinearFunctional(lfs::Vector{<:AbstractLinearFunctional})
    return StackedLinearFunctional(Tuple(lfs)...)
end

"""
    output_shape(stacked::StackedLinearFunctional)

Returns the output shape of the stacked linear functional, which is the sum of the
output lengths of all component functionals.
"""
function output_shape(stacked::StackedLinearFunctional)
    total_length = mapreduce(lf -> prod(output_shape(lf)), +, stacked.linfunctionals)
    return (total_length,)
end
