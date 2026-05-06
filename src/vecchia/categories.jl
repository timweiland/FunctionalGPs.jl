export FunctionalCategory,
    INTEGRAL, FACE_INTEGRAL, EVALUATION, DERIVATIVE, OTHER,
    functional_category

"""
    FunctionalCategory

Coarse-grained classification of an `AbstractLinearFunctional` for the purpose
of choosing a Vecchia / sparse-Cholesky ordering. Categories are ranked from
coarsest (`INTEGRAL`) to finest (`DERIVATIVE`).

| Value | Meaning |
|-------|---------|
| `INTEGRAL` | Pure Lebesgue integrals over cells / boxes (coarsest). |
| `FACE_INTEGRAL` | Mixed evaluation × integral (e.g. axis-aligned line integrals). |
| `EVALUATION` | Point evaluation. |
| `DERIVATIVE` | Composition with a differential operator (finest). |
| `OTHER` | Anything not recognised. |
"""
@enum FunctionalCategory begin
    INTEGRAL
    FACE_INTEGRAL
    EVALUATION
    DERIVATIVE
    OTHER
end

"""
    functional_category(L::AbstractLinearFunctional) -> FunctionalCategory

Classify a linear functional. Used by `vecchia` to decide a sensible
block-ordering for the sparse Cholesky.

Recognised cases:
- [`EvaluationFunctional`](@ref) → `EVALUATION`.
- [`VectorizedLebesgueIntegral`](@ref) → `INTEGRAL`.
- [`LinFctlLinFuncOpConcat`](@ref) wrapping a [`PartialDerivative`](@ref) →
  `DERIVATIVE`; otherwise delegates to the inner functional.
- [`TensorProductFunctional`](@ref) → `FACE_INTEGRAL` if it mixes evaluation
  with integration, `INTEGRAL` if all factors are integrals, else `EVALUATION`.
- [`SumLinearFunctional`](@ref) → the finest category among its summands.

Unknown types fall back to `OTHER`.
"""
functional_category(::EvaluationFunctional) = EVALUATION
functional_category(::VectorizedLebesgueIntegral) = INTEGRAL

function functional_category(L::LinFctlLinFuncOpConcat)
    for op in L.linfuncops
        op isa PartialDerivative && return DERIVATIVE
    end
    return functional_category(L.linfctl)
end

function functional_category(L::TensorProductFunctional)
    cats = map(functional_category, L.factors)
    has_eval = EVALUATION in cats
    has_integral = INTEGRAL in cats
    if has_eval && has_integral
        return FACE_INTEGRAL
    elseif all(==(INTEGRAL), cats)
        return INTEGRAL
    else
        return EVALUATION
    end
end

function functional_category(L::SumLinearFunctional)
    cats = map(functional_category, summands(L))
    DERIVATIVE in cats && return DERIVATIVE
    EVALUATION in cats && return EVALUATION
    FACE_INTEGRAL in cats && return FACE_INTEGRAL
    INTEGRAL in cats && return INTEGRAL
    return OTHER
end

functional_category(::AbstractLinearFunctional) = OTHER
