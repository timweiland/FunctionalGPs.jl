using FunctionalGPs
using Pkg
using Coverage

Pkg.test("FunctionalGPs"; coverage = true, test_args = ["skip-aqua"])

coverage = process_folder()
LCOV.writefile("lcov.info", coverage)
clean_folder(".")
