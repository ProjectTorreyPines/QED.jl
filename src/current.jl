function Jt_R(QI::QED_state; ι::FE_rep=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)# where{T<:Real, U<:AbstractVector{T}}
    J = zero(ρ)
    return Jt_R!(J, QI; ι, ρ)
end

function Jt_R!(J::AbstractVector{<:Real}, QI::QED_state; ι::FE_rep=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)# where {T<:Real, U<:AbstractVector{T}}
    χ(x) = x * ι(x) * fsa_∇ρ²_R²(QI, x)
    dχ(x) = ForwardDiff.derivative(χ, x)

    for (k, x) in enumerate(ρ)
        if x == 0
            γ = 1.0
        else
            γ = x * D(QI.dV_dρ, x) / QI.dV_dρ(x)
        end
        J[k] = dχ(x) + ι(x) * fsa_∇ρ²_R²(QI, x) * γ
    end
    J .*= QI.B₀ * QI.dΡ_dρ^2 / μ₀
    return J
end

function JB(QI::QED_state; ι=QI.ι)
    ρ = QI.ρ
    fsa_JB = QI.F.(ρ) .* Jt_R(QI, ι=ι)
    return fsa_JB .- D.(Ref(QI.F), ρ) .* dΦ_dρ.(Ref(QI), ρ) .* fsa_∇ρ²_R².(Ref(QI), ρ) .* ι.(ρ) ./ (2π * μ₀)
end

function Ip(QI::QED_state)
    return dΦ_dρ(QI, 1) * QI.dV_dρ(1) * fsa_∇ρ²_R²(QI, 1) * QI.ι(1) / ((2π)^2 * μ₀)
end

function Vni(QI::QED_state, η, x)
    return 2π * η(x) * QI.JBni(x) / (QI.F(x) * QI.fsa_R⁻²(x))
end