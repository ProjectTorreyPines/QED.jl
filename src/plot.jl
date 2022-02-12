function plot_JtoR_profiles(QI::QED_state, ιs::Vector{FE_rep}, times::Vector{Float64}, Ncol::Int)
    ρ = QI.ρ
    p = Plots.plot(legend=:best, color_palette=Plots.palette(:plasma, Ncol))
    Plots.plot!(ρ, QI.JtoR.(ρ), marker=:circle, color=:black, label="Real initial <Jt/R>")
    Plots.plot!(ρ, Jt_R(QI), label="t=$(0.0)", linewidth=2)
    for (n, ι) in enumerate(ιs)
        Plots.plot!(ρ, Jt_R(QI, ι=ι), linewidth=2, label="t=$(times[n])")
    end
    display(p)
end