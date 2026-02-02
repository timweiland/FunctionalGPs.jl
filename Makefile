.PHONY: format test docs docs-serve docs-logo clean-cov

format: ## Format code using Runic
	@runic --inplace .

test: ## Run full test suite
	julia --project=. -e "using Pkg; Pkg.test()"

test-cov: ## Run full test suite with coverage
	julia --project=. -e "using Pkg; Pkg.test(; coverage=true)"
	julia --project=. -e "using Coverage; coverage = process_folder(); LCOV.writefile(\"coverage-lcov.info\", coverage)"

docs-logo: ## Generate logo from GP computations
	julia --project=docs docs/src/assets/generate_logo.jl

docs: docs-logo ## Generate documentation (rebuilds logo first)
	julia --project=docs docs/make.jl

docs-serve: ## Serve documentation locally with VitePress
	cd docs && npm run docs:dev

clean-cov: ## Clean up coverage files
	find . -name "*.jl.*.cov" -delete
