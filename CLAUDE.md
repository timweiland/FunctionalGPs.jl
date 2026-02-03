# FunctionalGPs.jl

A Julia package for Gaussian process regression with mixed observations: point evaluations, derivatives, and integrals.

## Quick Commands

```bash
# Format code (uses Runic)
make format

# Run tests
make test

# Run tests with coverage
make test-cov

# Build documentation
make docs

# Serve docs locally
make docs-serve
```

### Alternative test commands

```bash
# Skip Aqua checks (faster iteration)
julia --project -e 'using Pkg; Pkg.test(test_args=["skip-aqua"])'

# Run specific test pattern with ReTest (edit test/runtests.jl or use REPL)
julia --project -e 'using ReTest, FunctionalGPs; include("test/FunctionalGPsTests.jl"); FunctionalGPsTests.retest(r"pattern")'
```

## Architecture

The codebase follows a layered architecture (see `src/FunctionalGPs.jl`):

1. **Utilities** (`util/`) - Array ops, Cholesky helpers, Kronecker utilities
2. **Layer 1: Building blocks**
   - `domains/` - Intervals, boxes, grids
   - `kernels/` - Kernel definitions (Matern, Wendland, derivatives)
   - `operators/` - Linear operators (partial derivatives, concatenation, scaling)
   - `functionals/` - Linear functionals (evaluation, integrals)
3. **Layer 2: Intermediate representation** (`crosscov/`) - Cross-covariance computations
4. **Layer 3: Output matrices** (`matrices/`) - Covariance matrix construction
5. **Specializations** (`specializations/`) - Trait-dispatched implementations for specific kernels (Matern, compact, stationary)
6. **Composition** (`composition/`) - How layers connect (functional kernels, operator kernels)
7. **High-level APIs** (`gps/`, `problems/`) - GP conditioning, problem-specific interfaces (heat equation)

## Code Style

- **Formatter**: Runic (run `make format`)
- **Naming**: `CamelCase` for types/modules, `snake_case` for functions/variables
- **Margin**: 90 characters
- **Style rules** (from `.JuliaFormatter.toml`):
  - `always_use_return = true`
  - `separate_kwargs_with_semicolon = true`
  - `always_for_in = true`

## Testing

- Framework: ReTest + Test + Aqua (API hygiene)
- Tests mirror source structure: `src/kernels/foo.jl` → `test/kernels/test_foo.jl`
- Test aggregator: `test/FunctionalGPsTests.jl`
- CI runs on Julia 1.10 and nightly

## Project Structure

```
src/
├── FunctionalGPs.jl      # Entry point
├── domains/              # Interval, Box, FactorizedBox, grids
├── kernels/              # Matern, Wendland, derivatives, traits
├── operators/            # PartialDerivative, LinearDiffOp, Scale, Sum
├── functionals/          # Evaluation, LebesgueIntegral, Sum, Stack
├── crosscov/             # Cross-covariance computations
├── matrices/             # CovarianceMatrix construction
├── specializations/      # Kernel-specific optimizations
├── composition/          # Functional/operator kernel composition
├── gps/                  # GP conditioning, observations
├── problems/             # Heat equation, IBVPs
└── util/                 # Helper functions

test/
├── runtests.jl           # Entry point
├── FunctionalGPsTests.jl # Test module aggregator
└── [mirrors src/]        # Test files
```

## Key Dependencies

- `AbstractGPs` - Base GP interface
- `KernelFunctions` - Kernel primitives
- `Kronecker` - Kronecker product operations
- `LinearOperators` - Linear operator abstractions
