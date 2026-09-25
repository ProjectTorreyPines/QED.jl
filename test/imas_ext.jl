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
