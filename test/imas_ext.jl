using IMAS

@testset "IMAS native current grid" begin
    file = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z26-3000.json")
    dd = IMAS.json2imas(file; show_warnings=false)
    eq = dd.equilibrium.time_slice[].profiles_1d
    cp = dd.core_profiles.profiles_1d[]

    rho = collect(range(0.0, 1.0; length=37))
    j_tor = 1.5e6 .* (1 .- 0.75 .* rho .^ 2) .* (1 .+ 0.12 .* sin.(9π .* rho))
    cp.grid.rho_tor_norm = rho
    cp.j_tor = j_tor

    gm9 = IMAS.interp1d(IMAS.norm01(eq.rho_tor), eq.gm9, :cubic).(rho)
    state = QED.initialize(dd; uniform_rho=51, ip_from=1.0e6)
    reconstructed = state.JtoR.(rho) ./ gm9
    @test reconstructed ≈ j_tor rtol=1e-12

    equilibrium_state = QED.initialize(dd; uniform_rho=51, j_tor_from=:equilibrium, ip_from=1.0e6)
    @test equilibrium_state.JtoR.(IMAS.norm01(eq.rho_tor)) ≈ eq.j_tor .* eq.gm9 rtol=1e-12
end

@testset "IMAS current round trips and evolution operator" begin
    file = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z26-3000.json")
    dd = IMAS.json2imas(file; show_warnings=false)
    eqt = dd.equilibrium.time_slice[]
    eq = eqt.profiles_1d
    cp = dd.core_profiles.profiles_1d[]
    B0 = eqt.global_quantities.vacuum_toroidal_field.b0
    rho = collect(range(0.0, 1.0; length=37))
    cp.grid.rho_tor_norm = rho
    # Supply a smooth, nontrivial parallel/toroidal conversion on the public
    # sample geometry. This fixture tests transfer, not equilibrium accuracy.
    eq.gm5 = eq.f .^ 2 .* eq.gm1 .* (1 .+ 0.1 .* eq.rho_tor_norm .^ 2)
    eq.dpressure_dpsi = -1.0e4 .* (1 .- eq.rho_tor_norm .^ 2)
    jtor = 1.5e6 .* (1 .- 0.75 .* rho .^ 2) .* (1 .+ 0.12 .* sin.(9π .* rho))
    cp.j_tor = jtor
    cp.j_total = IMAS.Jtor_2_Jpar(rho, jtor, true, eqt)
    initial_parallel = copy(cp.j_total)
    initial_equilibrium = deepcopy(IMAS.imas2dict(dd.equilibrium))
    gm9 = IMAS.interp1d(IMAS.norm01(eq.rho_tor), eq.gm9, :cubic).(rho)

    state = QED.initialize(dd; uniform_rho=51, ip_from=1.0e6)
    @test QED.Jt_R(state; ρ=rho) ./ gm9 ≈ jtor rtol=1e-12
    @test QED.JB(state; ρ=rho) ./ B0 ≈ initial_parallel rtol=1e-12
    for _ in 1:100
        current = QED.initialize(dd; uniform_rho=51, ip_from=1.0e6)
        cp.j_total = QED.JB(current; ρ=rho) ./ B0
        cp.j_tor = IMAS.Jpar_2_Jtor(rho, cp.j_total, true, eqt)
    end
    @test cp.j_total ≈ initial_parallel rtol=1e-12
    @test cp.j_tor ≈ jtor rtol=1e-12
    @test isequal(IMAS.imas2dict(dd.equilibrium), initial_equilibrium)

    # A changed transform must change the current. The voltage used by the
    # diffusion matrix must equal Ohm's law for the returned parallel current,
    # including off-grid positions and the magnetic axis.
    iota = QED.FE(state.ρ, state.ι.(state.ρ) .* (1 .+ 0.03 .* state.ρ .^ 2))
    sample_rho = [0.0, 0.003, 0.07, 0.31, 0.79, 1.0]
    eta(x) = 1.0e-8 * (1 + x^2)
    tiny_step = QED.diffuse(state, eta, 1.0e-10, 1; Ip=QED.Ip(state))
    @test QED.JB(tiny_step; ρ=rho) ./ B0 ≈ initial_parallel rtol=1e-7
    evolved = QED.QED_state(state, iota, QED.Jt_R(state; ι=iota))
    current = QED.JB(evolved; ρ=sample_rho)
    voltage = [QED.αβ(x, evolved, eta) * QED.D(iota, x) +
               QED.αdβ_dρ(x, evolved, eta) * iota(x) for x in sample_rho]
    expected = [2π * eta(x) * j / (evolved.F(x) * evolved.fsa_R⁻²(x))
                for (x, j) in zip(sample_rho, current)]
    @test voltage ≈ expected rtol=1e-12
    @test !isapprox(current, QED.JB(state; ρ=sample_rho); rtol=1e-4)
    with_source = QED.QED_state(evolved; JBni=x -> 1e5)
    @test QED.JB(with_source; ρ=sample_rho) ≈ current rtol=1e-12
    @test QED.Ip(evolved) ≈ 1.03 * QED.Ip(state) rtol=1e-12
end

@testset "Uniform cylindrical current with current reference" begin
    rho = collect(range(0.0, 1.0; length=41))
    R0, B0, j = 3.0, 2.0, 1.0e6
    iota = QED.μ₀ * R0 * j / (2B0)
    initial = QED.initialize(rho, B0, fill(1 / R0^2, length(rho)),
                             fill(R0 * B0, length(rho)), 4π^2 * R0 .* rho,
                             fill(1 / iota, length(rho)), fill(j, length(rho)),
                             fill(1 / R0, length(rho));
                             ρ_j_tor=(rho, fill(j / R0, length(rho))),
                             ρ_j_parallel=(rho, fill(j, length(rho))))
    eta(x) = 1.0e-8
    target = π * j
    voltage = 2π * eta(0.0) * R0 * j
    @test QED.Ip(initial) ≈ target rtol=1e-12
    for result in (QED.diffuse(initial, eta, 0.1, 10; Ip=target),
                   QED.steady_state(initial, eta; Ip=target),
                   QED.steady_state(initial, eta; Vedge=voltage))
        @test QED.JB(result; ρ=rho) ./ B0 ≈ fill(j, length(rho)) rtol=1e-8
        @test QED.Jt_R(result; ρ=rho) .* R0 ≈ fill(j, length(rho)) rtol=1e-8
        @test QED.Ip(result) ≈ target rtol=1e-8
    end
    # Non-inductive current enters the same loop-voltage balance.
    driven = QED.QED_state(initial; JBni=x -> 0.3B0*j)
    result = QED.steady_state(driven, eta; Vedge=0.7voltage)
    @test QED.JB(result; ρ=rho) ./ B0 ≈ fill(j, length(rho)) rtol=1e-8
end

@testset "Cylindrical current eigenmode decay" begin
    # Power series for J_0 and J_1, evaluated only for |x| <= 3.832.
    function bessel(n, x)
        term = n == 0 ? 1.0 : x / 2
        value = term
        for m in 1:20
            term *= -x^2 / (4m * (m + n))
            value += term
        end
        return value
    end

    R0, B0, j0, amplitude = 3.0, 2.0, 1.0e6, 0.2
    k = 3.8317059702075125 # First positive zero of J_1: fixed total current.
    eta(x) = 1.0e-8
    duration = 2.0
    errors = Float64[]
    for n in (41, 81, 161)
        rho = collect(range(0.0, 1.0; length=n))
        current = j0 .* (1 .+ amplitude .* bessel.(0, k .* rho))
        iota = [QED.μ₀ * R0 * j0 / (2B0) *
                (1 + (x == 0 ? amplitude : 2amplitude*bessel(1, k*x)/(k*x))) for x in rho]
        initial = QED.initialize(rho, B0, fill(1/R0^2, n), fill(R0*B0, n),
                                 4π^2*R0 .* rho, 1 ./ iota, current, fill(1/R0, n);
                                 ρ_j_tor=(rho, current ./ R0), ρ_j_parallel=(rho, current))
        result = QED.diffuse(initial, eta, duration, 2000; Ip=π*j0)
        # j_t = (eta/mu0) * (j'' + j'/rho) in a cylinder of unit minor radius.
        exact = j0 .* (1 .+ amplitude .* exp(-eta(0.0)/QED.μ₀*k^2*duration) .* bessel.(0, k .* rho))
        computed = QED.JB(result; ρ=rho) ./ B0
        push!(errors, sqrt(sum(abs2, computed - exact) / sum(abs2, exact)))
        @test QED.Ip(result) ≈ π*j0 rtol=1e-12
        @test QED.Jt_R(result; ρ=rho) .* R0 ≈ computed rtol=1e-12
    end
    @test errors[1] < 2e-6
    @test errors[2] < errors[1] / 10
    @test errors[3] < errors[2] / 10
end
