using GaussPDE
using FiniteDifferences

@testset "Heat1DIBVPTruncatedSineICDirichletBC" begin
    box_domain = BoxDomain((0.0, 2.0), (0.0, 1.0))
    c = 0.25
    ρ = 0.75
    κ = 0.5
    ic_coeffs = [1.0, 2.0, 3.0]

    p = Heat1DIBVPTruncatedSineICDirichletBC(
        box_domain;
        c = c,
        ρ = ρ,
        κ = κ,
        ic_coeffs = ic_coeffs,
    )

    @test domain(p) == box_domain
    ℒs = lindiffops(p)
    @test length(ℒs) == 1
    ℒ = ℒs[1]
    @test ℒ isa LinearDifferentialOperator{2, 1, 2}
    @test ℒ.idx_dict[1] == Dict((1, 0) => c * ρ, (0, 2) => -κ)

    x = [0.5, 0.6, 0.7]
    @test ic_fn(p, x) ≈
        ic_coeffs[1] * sin.(π .* x) +
        ic_coeffs[2] * sin.(2π .* x) +
        ic_coeffs[3] * sin.(3π .* x)

    N = 10
    noise = 1.0e-8
    ic_observation = sample_ic(p, N; noise = noise)
    @test ic_observation.ℒ isa EvaluationFunctional
    @test length(ic_observation.ℒ.X) == N
    @test length(ic_observation.y) == N
    @test !isnothing(ic_observation.ε)

    bc_observation = sample_bc(p, N; noise = noise)
    @test length(bc_observation.ℒ.X) == 2 * N
    @test size(bc_observation.y) == (N, 2)
    @test !isnothing(bc_observation.ε)

    sol = solution(p)

    dt_fd = (t, x) -> central_fdm(12, 1)((τ) -> sol(τ, x), t)
    dx²_fd = (t, x) -> central_fdm(12, 2)((χ) -> sol(t, χ), x)
    heat_op_fd(t, x) = c * ρ * dt_fd(t, x) - κ * dx²_fd(t, x)

    ts = range(0.05; stop = 2.0 - 1.0e-2, length = 10)
    xs = range(0.0 + 1.0e-2; stop = 1.0 - 1.0e-2, length = 10)
    @testset "Solution satisfies heat equation (t=$t, x=$x)" for t in ts, x in xs
        @test heat_op_fd(t, x) ≈ 0.0 atol = 1.0e-4
    end
end
