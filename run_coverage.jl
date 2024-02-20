using GaussPDE
using Pkg
using Coverage

Pkg.test("GaussPDE"; coverage=true, test_args=["skip-aqua"])

coverage = process_folder()
LCOV.writefile("lcov.info", coverage)
clean_folder(".")
