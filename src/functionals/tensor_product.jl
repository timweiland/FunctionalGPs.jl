import KernelFunctions: ⊗

export TensorProductFunctional, factors

"""
    TensorProductFunctional{N, F}

A tensor product of N linear functionals, where each functional operates on
a separate dimension of a tensor product kernel.

For a tensor product kernel `k = k₁ ⊗ k₂ ⊗ ... ⊗ kₙ`, applying a
`TensorProductFunctional` with functionals `ℒ = ℒ₁ ⊗ ℒ₂ ⊗ ... ⊗ ℒₙ`
produces `ℒ₁(k₁) ⊗ ℒ₂(k₂) ⊗ ... ⊗ ℒₙ(kₙ)`.

This is particularly useful for axis-aligned line integrals on 2D domains:
```julia
k = k_x ⊗ k_y
eval_x = EvaluationFunctional(x_points)
∫_y = VectorizedLebesgueIntegral(y_intervals)
ℒ = eval_x ⊗ ∫_y  # Line integral: evaluate at x points, integrate over y
```

# Type Parameters
- `N`: Number of factors
- `F <: NTuple{N, AbstractLinearFunctional}`: Concrete tuple type for type stability
"""
struct TensorProductFunctional{N, F <: NTuple{N, AbstractLinearFunctional}} <: AbstractLinearFunctional
    factors::F

    function TensorProductFunctional(factors::F) where {N, F <: NTuple{N, AbstractLinearFunctional}}
        return new{N, F}(factors)
    end
end

"""
    factors(ℒ::TensorProductFunctional)

Return the tuple of factor functionals.
"""
factors(ℒ::TensorProductFunctional) = ℒ.factors

"""
    TensorProductFunctional(factors...)

Construct a TensorProductFunctional from individual functionals.
"""
function TensorProductFunctional(factors::AbstractLinearFunctional...)
    return TensorProductFunctional(factors)
end

"""
    output_shape(ℒ::TensorProductFunctional)

Return the combined output shape, which is the concatenation of individual shapes.
"""
function output_shape(ℒ::TensorProductFunctional)
    return Tuple(Iterators.flatten(output_shape.(ℒ.factors)))
end

function Base.show(io::IO, ℒ::TensorProductFunctional)
    return print(io, join(["$(factor)" for factor in factors(ℒ)], " ⊗ "))
end

# ⊗ operator for composing functionals

function ⊗(ℒ₁::AbstractLinearFunctional, ℒ₂::AbstractLinearFunctional)
    return TensorProductFunctional((ℒ₁, ℒ₂))
end

function ⊗(ℒ₁::TensorProductFunctional, ℒ₂::AbstractLinearFunctional)
    return TensorProductFunctional((factors(ℒ₁)..., ℒ₂))
end

function ⊗(ℒ₁::AbstractLinearFunctional, ℒ₂::TensorProductFunctional)
    return TensorProductFunctional((ℒ₁, factors(ℒ₂)...))
end

function ⊗(ℒ₁::TensorProductFunctional, ℒ₂::TensorProductFunctional)
    return TensorProductFunctional((factors(ℒ₁)..., factors(ℒ₂)...))
end
