using QED
import QED: JSON
using Test
using Plots

μ₀ = 4e-7 * π

@testset "TRANSP-bm_fixed-Vedge" begin

    # Load TRANSP data at 3.0 s
    file_0 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z26-3000.json")
    transp_0 = JSON.parsefile(file_0)
    QI_0 = from_imas(transp_0)
    η = η_imas(transp_0)

    # Diffuse for 1.0 s (note: θimp is increased to 0.5 for stability with the coarse time step)
    QI = diffuse(QI_0, η, 1.0, 10000; θimp=0.5, Np=1000, Vedge=0.0, debug=true)

    # Compare to TRANSP data at 4.0 s
    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z26-4000.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    ρ = QI_0.ρ

    p = plot(; title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ); label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    p = plot(; title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ); marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ); marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI); label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    rtol = 1e-6
    @test isapprox(transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], 1.0 ./ QI_1.ι.(ρ), rtol=rtol)

    Ip0 = transp_0["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    Ip1 = transp_1["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    rtol = 2 * abs((Ip0 - Ip(QI_0)) / Ip0)
    @test isapprox(Ip1, Ip(QI_1), rtol=rtol)
end

@testset "TRANSP-bm_fixed-current" begin

    # Load TRANSP data at 2.91 s
    file_0 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z27-2910.json")
    transp_0 = JSON.parsefile(file_0)
    QI_0 = from_imas(transp_0)
    η = η_imas(transp_0)

    ρ = deepcopy(QI_0.ρ)

    #Compute matrices
    T = QED.define_T(QI_0)
    Y = QED.define_Y(QI_0, η)

    # Diffuse for 1.0 s with current held fixed
    QI = QED._diffuse(QI_0, η, 1.0, 10000, T, Y; Np=1000)

    # Diffuse for 1.0 s with current held fixed
    QI_2MA = QED._diffuse(QI_0, η, 1.0, 10000, T, Y; θimp=0.75, Ip=2e6, Np=1000)

    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z27-3910.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    p = plot(; title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ); label="QED end", linewidth=3, color=:deepskyblue)
    plot!(ρ, -1.0 ./ QI_2MA.ι.(ρ); label="QED 2 MA", linewidth=3, color=:cyan)
    display(p)

    p = plot(; title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ); marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ); marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0; ρ=ρ); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI; ρ=ρ); label="QED end", linewidth=3, color=:deepskyblue)
    plot!(ρ, Jt_R(QI_2MA; ρ=ρ); label="QED 2 MA", linewidth=3, color=:cyan)
    display(p)

    @test QI_0.ι(1) ≈ QI.ι(1)

    Ip0 = transp_0["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    Ip1 = transp_1["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    rtol = 2 * abs((Ip0 - Ip(QI_0)) / Ip0)
    @test isapprox(Ip1, Ip(QI), rtol=rtol)

    @test Ip(QI_2MA) ≈ 2e6
end

@testset "TRANSP-bm_NI-Ip" begin

    # Load TRANSP data at 2.91 s
    file_0 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z28-2910.json")
    transp_0 = JSON.parsefile(file_0)
    QI_0 = from_imas(transp_0)
    η = η_imas(transp_0)

    # Diffuse for 1.0 s with current held fixed
    QI = diffuse(QI_0, η, 1.0, 10000; Np=1000)

    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z28-3910.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    ρ = QI_0.ρ

    p = plot(; title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ); label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    p = plot(; title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ); marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ); marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI); label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    @test QI_0.ι(1) ≈ QI.ι(1)

    Ip0 = transp_0["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    Ip1 = transp_1["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    rtol = 2 * abs((Ip0 - Ip(QI_0)) / Ip0)
    @test isapprox(Ip1, Ip(QI), rtol=rtol)

end

@testset "TRANSP-bm_NI-Vedge" begin

    # Load TRANSP data at 2.91 s
    file_0 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z29-2910.json")
    transp_0 = JSON.parsefile(file_0)
    QI_0 = from_imas(transp_0)
    η = η_imas(transp_0)

    # Diffuse for 1.0 s with current held fixed
    QI = diffuse(QI_0, η, 1.0, 10000; Np=1000, Vedge=0.1)

    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z29-3910.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    ρ = QI_0.ρ

    p = plot(; title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"]; marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ); label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    p = plot(; title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ); marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ); marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0); label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI); label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    # Passes eye test
    Ip0 = transp_0["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    Ip1 = transp_1["equilibrium"]["time_slice"][1]["global_quantities"]["ip"]
    rtol = 1.2e-2
    @test isapprox(Ip1, Ip(QI), rtol=rtol)

end

@testset "steady-state_NI" begin

    # Load TRANSP data at 3.0 s
    file_0 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z26-3000.json")
    transp_0 = JSON.parsefile(file_0)
    QI_0 = from_imas(transp_0)
    η = η_mock()

    JBni(x) = -1e6 * (0.9 * sin(2π * x) + 0.1)
    QI_0 = QED.QED_state(QI_0; JBni=JBni)

    Y = QED.define_Y(QI_0, η)

    ρ = QI_0.ρ

    p = plot(; title="<J⋅B>", legend=:bottomleft)
    plot!(ρ, JBni.(ρ); label="Target", marker=:circle, color=:black)
    plot!(ρ, JB(QI_0); label="QED start", linewidth=3, color=:blue)

    # Diffuse for 5.0 s at a time
    color = [:purple4, :purple3, :purple1]
    QI_ev = deepcopy(QI_0)
    for (i, col) in enumerate(color)
        QI_ev = diffuse(QI_ev, η, 5.0, 500; Vedge=0.0)
        plot!(ρ, JB(QI_ev); label="QED $(i * 10) s", linewidth=3, color=col)
    end

    QI_ss = steady_state(QI_0, η; Vedge=0.0)
    QI_ss = QED._steady_state(QI_0, η, Y; Vedge=0.0)
    plot!(ρ, JB(QI_ss); label="QED steady-state", linewidth=3, color=:deepskyblue)

    display(p)

    @test isapprox(JB(QI_ss), JBni.(ρ), rtol=1e-2)
end

@testset "Waveform" begin

    @testset "constant" begin
        W = QED.Waveform(3.5)
        @test W isa QED.Waveform{Float64}
        @test W(0.0) == 3.5
        @test W(-1e6) == 3.5
        @test W(1e6) == 3.5
    end

    @testset "interpolates knots exactly" begin
        ts = collect(range(0.0, 100.0, 21))
        vs = @. 1.0e3 * (1.0 - exp(-ts / 20.0)) * cos(ts / 30.0)
        W = QED.Waveform(ts, vs)
        @test maximum(abs, [W(t) - v for (t, v) in zip(ts, vs)]) < 1e-9 * maximum(abs, vs)
    end

    @testset "non-uniform grid" begin
        ts = [0.0, 0.5, 3.0, 3.1, 8.0, 20.0]
        vs = [1.0, -2.0, 0.5, 0.6, -3.0, 4.0]
        W = QED.Waveform(ts, vs)
        @test maximum(abs, [W(t) - v for (t, v) in zip(ts, vs)]) < 1e-9 * maximum(abs, vs)
    end

    # A natural cubic spline (S''= 0 at both ends) reproduces linear data exactly, and
    # `Extension` continues the boundary cubic, so extrapolation stays exactly linear too.
    # This pins down both the boundary condition and the extrapolation rule analytically.
    #
    # Range note: the spline solve leaves ~1e-18 of roundoff in the cubic coefficients, and
    # the extension amplifies it by (t - t_end)^3. Staying within a few multiples of the data
    # span keeps that below 1e-9; at 100x the span it grows to ~1e-9 relative. That is inherent
    # to a natural cubic, not to the backend (DataInterpolations shows the same magnitude).
    @testset "linear data is exact, inside and outside" begin
        a, b = -7.0, 0.35
        ts = collect(range(0.0, 100.0, 11))
        W = QED.Waveform(ts, @. a + b * ts)
        for t in (-500.0, -1.0, 0.0, 17.3, 100.0, 250.0, 600.0)
            @test isapprox(W(t), a + b * t; rtol=1e-9, atol=1e-8)
        end
    end

    @testset "natural boundary condition" begin
        ts = collect(range(0.0, 10.0, 11))
        vs = sin.(ts)
        W = QED.Waveform(ts, vs)
        d2(t, h) = (W(t + h) - 2W(t) + W(t - h)) / h^2
        h = 1e-3
        @test abs(d2(ts[1], h)) < 1e-6
        @test abs(d2(ts[end], h)) < 1e-6
        # sanity: the spline is genuinely curved in between, so the check above has teeth
        @test abs(d2(ts[6], h)) > 0.1
    end

    @testset "extension is not clamping" begin
        ts = collect(range(0.0, 10.0, 11))
        W = QED.Waveform(ts, sin.(ts))
        @test W(-2.0) != W(ts[1])
        @test W(12.0) != W(ts[end])
        @test isfinite(W(-2.0)) && isfinite(W(12.0))
    end

    @testset "promotion and type stability" begin
        ts = collect(range(0.0, 10.0, 11))
        @test QED.Waveform(ts, collect(1:11)) isa QED.Waveform{Float64}
        @test QED.Waveform(collect(0:10), sin.(0:10)) isa QED.Waveform{Float64}
        W = QED.Waveform(ts, sin.(ts))
        @test (@inferred W(1.5)) isa Float64
    end

    @testset "drives QED_build voltages" begin
        Nc = 4
        ts = collect(range(0.0, 10.0, 11))
        waveforms = [QED.Waveform(ts, @. 1.0e3 * sin(ts / 3 + k)) for k in 1:Nc]
        Ic = zeros(Nc)
        Vc = zeros(Nc)
        Rc = fill(1e-3, Nc)
        Mcc = [i == j ? 1.0e-5 : 2.0e-6 for i in 1:Nc, j in 1:Nc]  # diagonally dominant, invertible
        build = QED.QED_build(Ic, Vc, Rc, Mcc, waveforms)

        QED.update_voltages!(build, 4.2)
        @test build.Vc ≈ [w(4.2) for w in waveforms]

        # coil-circuit evolution runs and conserves nothing exotic, just stays finite
        Is = QED.evolve!(build, 10.0, 100)
        @test size(Is) == (Nc, 101)
        @test all(isfinite, Is)
    end
end
