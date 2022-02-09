function Jt_R(QI::QED_state; ι=nothing)
    ι === nothing && (ι = QI.ι)

    χ(x) = x * ι(x) * QI.fsa_∇ρ²_R²(x)
    dχ(x) = ForwardDiff.derivative(χ, x)

    ρ = QI.ρ
    γ = zero(ρ)
    for k in 1:length(ρ)
        if ρ[k] == 0
            γ[k] = 1.0
        else
            γ[k] = ρ[k] * D(QI.dV_dρ, ρ[k])/ QI.dV_dρ(ρ[k])
        end
    end
    return QI.B₀ * QI.dΡ_dρ^2 * (dχ.(ρ) + ι.(ρ) .* QI.fsa_∇ρ²_R².(ρ) .* γ) / μ₀
end

function JB(QI::QED_state; ι=nothing)
    ι === nothing && (ι = QI.ι)
    ρ = QI.ρ
    fsa_JB = QI.F.(ρ) .* Jt_R(QI, ι=ι) 
    return fsa_JB .- D.(Ref(QI.F), ρ) .* QI.dΦ_dρ.(ρ) .* QI.fsa_∇ρ²_R².(ρ) .* ι.(ρ) ./ (2π * μ₀)
end

function Ip(QI::QED_state)
    return QI.dΦ_dρ(1) * QI.dV_dρ(1) * QI.fsa_∇ρ²_R²(1) * QI.ι(1) / ((2π)^2 * μ₀)
end

function Vni(QI::QED_state, η, x) 
    return 2π * η(x) * QI.JBni(x) / (QI.F(x) * QI.fsa_R⁻²(x))
end