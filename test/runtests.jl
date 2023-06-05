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

    # Diffuse for 1.0 s
    QI = diffuse(QI_0, η, 1.0, 10000, θimp=0.25, Np=1000, Vedge=0.0, debug=true)

    # Compare to TRANSP data at 4.0 s
    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z26-4000.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    ρ = QI_0.ρ

    p = plot(title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ), label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    p = plot(title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ), marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ), marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI), label="QED end", linewidth=3, color=:deepskyblue)
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
    T = define_T(QI_0)
    Y = define_Y(QI_0, η)

    # Diffuse for 1.0 s with current held fixed
    QI = diffuse(QI_0, η, 1.0, 10000, T, Y, Np=1000)

    # Diffuse for 1.0 s with current held fixed
    QI_2MA = diffuse(QI_0, η, 1.0, 10000, T, Y, θimp=0.75, Ip=2e6, Np=1000)

    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z27-3910.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    p = plot(title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ), label="QED end", linewidth=3, color=:deepskyblue)
    plot!(ρ, -1.0 ./ QI_2MA.ι.(ρ), label="QED 2 MA", linewidth=3, color=:cyan)
    display(p)

    p = plot(title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ), marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ), marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0, ρ=ρ), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI, ρ=ρ), label="QED end", linewidth=3, color=:deepskyblue)
    plot!(ρ, Jt_R(QI_2MA, ρ=ρ), label="QED 2 MA", linewidth=3, color=:cyan)
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
    QI = diffuse(QI_0, η, 1.0, 10000, Np=1000)

    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z28-3910.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    ρ = QI_0.ρ

    p = plot(title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ), label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    p = plot(title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ), marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ), marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI), label="QED end", linewidth=3, color=:deepskyblue)
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
    QI = diffuse(QI_0, η, 1.0, 10000, Np=1000, Vedge=0.1)

    file_1 = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_163303Z29-3910.json")
    transp_1 = JSON.parsefile(file_1)
    QI_1 = from_imas(transp_1)

    ρ = QI_0.ρ

    p = plot(title="Safety Factor", legend=:bottomleft)
    plot!(ρ, -transp_0["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, -transp_1["equilibrium"]["time_slice"][1]["profiles_1d"]["q"], marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, -1.0 ./ QI_0.ι.(ρ), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, -1.0 ./ QI.ι.(ρ), label="QED end", linewidth=3, color=:deepskyblue)
    display(p)

    p = plot(title="<Jt/R>", legend=:bottomleft)
    plot!(ρ, QI_0.JtoR.(ρ), marker=:circle, label="TRANSP start", color=:darkred)
    plot!(ρ, QI_1.JtoR.(ρ), marker=:circle, label="TRANSP end", color=:tomato)
    plot!(ρ, Jt_R(QI_0), label="QED start", linewidth=3, color=:blue)
    plot!(ρ, Jt_R(QI), label="QED end", linewidth=3, color=:deepskyblue)
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
    QI_0 = QED_state(QI_0, JBni=JBni)

    Y = define_Y(QI_0, η)

    ρ = QI_0.ρ

    p = plot(title="<J⋅B>", legend=:bottomleft)
    plot!(ρ, JBni.(ρ), label="Target", marker=:circle, color=:black)
    plot!(ρ, JB(QI_0), label="QED start", linewidth=3, color=:blue)

    # Diffuse for 5.0 s at a time
    color = [:purple4, :purple3, :purple1]
    QI_ev = deepcopy(QI_0)
    for (i, col) in enumerate(color)
        QI_ev = diffuse(QI_ev, η, 5.0, 500, Vedge=0.0)
        plot!(ρ, JB(QI_ev), label="QED $(i * 10) s", linewidth=3, color=col)
    end

    QI_ss = steady_state(QI_0, η, Vedge=0.0)
    QI_ss = steady_state(QI_0, η, Y, Vedge=0.0)
    plot!(ρ, JB(QI_ss), label="QED steady-state", linewidth=3, color=:deepskyblue)

    display(p)

    @test isapprox(JB(QI_ss), JBni.(ρ), rtol=1e-2)
end