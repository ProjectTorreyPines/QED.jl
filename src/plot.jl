function plot_JtoR_profiles(QI::QED_state, ιs::Vector{FE_rep}, times::Vector{Float64}, Ncol::Int)
    ρ = QI.ρ
    p = Plots.plot(; legend=:best, color_palette=Plots.palette(:plasma, Ncol))
    Plots.plot!(ρ, QI.JtoR.(ρ); marker=:circle, color=:black, label="Real initial <Jt/R>")
    Plots.plot!(ρ, Jt_R(QI); label="t=$(0.0)", linewidth=2)
    for (n, ι) in enumerate(ιs)
        Plots.plot!(ρ, Jt_R(QI; ι); linewidth=2, label="t=$(times[n])")
    end
    return display(p)
end

Plots.@recipe function plot_state(QI::QED_state; what=:q)
    ρ = QI.ρ
    Plots.@series begin
        label --> Ip(QI)
        if what == :JtoR
            ρ, QI.JtoR.(ρ)
        elseif what == :JB
            ρ, QED.JB(QI)
        elseif what == :JBni
            ρ, QI.JBni.(ρ)
        elseif what == :q
            ρ, abs.(1.0 ./ QI.ι.(ρ))
        else
            error("QED plot state can have `what=[:q, :JtoR, :JB, :JBni]`")
        end
    end
end
