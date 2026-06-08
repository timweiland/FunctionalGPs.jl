export Select

"""
    Select(p::Integer)

Linear operator that selects output `p` of a multi-output function. Composes with
the ordinary functionals via `∘`, so the spatial functionals stay output-agnostic:

```julia
EvaluationFunctional(X) ∘ Select(p)                       # output p at X
EvaluationFunctional(X) ∘ PartialDerivative((1,)) ∘ Select(p)  # ∂ of output p
```

Selecting a component of a vector-valued function is itself linear, so `Select`
slots into the operator algebra exactly like [`PartialDerivative`](@ref): it acts
on one kernel argument (`arg = 1` or `2`), pinning that argument's output index.
Applied to both arguments of a [`MultiOutputKernel`](@ref) it reduces the kernel
to the relevant single-output block (the zero kernel for distinct outputs of an
independent kernel — see [`BlockDiagonalKernel`](@ref)).
"""
struct Select <: AbstractLinearFunctionOperator
    output::Int
end

Base.show(io::IO, op::Select) = print(io, "Select(", op.output, ")")
