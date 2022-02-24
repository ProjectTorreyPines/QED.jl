struct QED_state{T <: Real}
    ρ::AbstractVector{T}
    dΡ_dρ::T
    B₀::T
    fsa_R⁻²::FE_rep{T}
    F::FE_rep{T}
    dV_dρ::FE_rep{T}
    ι::FE_rep{T}
    JtoR::FE_rep{T}
    dΦ_dρ
    dΨ_dρ
    fsa_∇ρ²_R²
    D_fsa_∇ρ²_R²
    JBni
end

function QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR; JBni=nothing,
                   x = 1.0 .- (1.0 .- range(0, 1, length=length(ρ))).^2, ε = 1e-3)

    dΦ_dρ(x) = 2π* B₀ * dΡ_dρ^2 * x
    dΨ_dρ(x) = ι(x) * dΦ_dρ(x)
    
    # If ξ = 2π*μ₀ * dV_dρ * <Jt/R> and χ = dV_dρ * dΨ_dρ * <|∇ρ|²/R²>
    # Then dχ/dρ = ξ
    # Solve for χ by integrating ξ, noting that χ=0 at ρ=0

    ξ = FE(x, 2π * μ₀ * dV_dρ.(x) .* JtoR.(x))

    C = zeros(2*length(x))
    C[2:2:end] .= I.(Ref(ξ), x) # value of χ is integral of ξ
    C[1:2:end] .= ξ.(x)         # derivative of χ is value of ξ
    χ = FE_rep(x, C)

    function fsa_∇ρ²_R²(r::Real)
        if r == 0
            # Linearly extrapolate to axis
            return 2*fsa_∇ρ²_R²(ε) - fsa_∇ρ²_R²(2ε)
        else
            return χ(r) / (dV_dρ(r) * dΨ_dρ(r))
        end
    end

    D_fsa_∇ρ²_R²(x) = ForwardDiff.derivative(fsa_∇ρ²_R², x)

    return QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR, dΦ_dρ, dΨ_dρ, fsa_∇ρ²_R², D_fsa_∇ρ²_R², JBni)
end

function QED_state(QI::QED_state, ι, JtoR)
    ι isa AbstractVector && (ι = FE(QI.ρ, ι))
    JtoR isa AbstractVector && (JtoR = FE(QI.ρ, JtoR))
    
    dΨ_dρ(x) = ι(x) * QI.dΦ_dρ(x)
    return QED_state(QI.ρ, QI.dΡ_dρ, QI.B₀, QI.fsa_R⁻², QI.F, QI.dV_dρ, ι, JtoR, QI.dΦ_dρ, dΨ_dρ, QI.fsa_∇ρ²_R², QI.D_fsa_∇ρ²_R², QI.JBni)
end

function QED_state(QI::QED_state;  JBni=nothing)
    JBni isa AbstractVector && (JBni = FE(QI.ρ, JBni))
    return QED_state(QI.ρ, QI.dΡ_dρ, QI.B₀, QI.fsa_R⁻², QI.F, QI.dV_dρ, QI.ι, QI.JtoR, QI.dΦ_dρ, QI.dΨ_dρ, QI.fsa_∇ρ²_R², QI.D_fsa_∇ρ²_R², JBni)
end

from_imas(filename::String, timeslice=1) = from_imas(JSON.parsefile(filename), timeslice)

function from_imas(data::Dict, timeslice=1)

    eqt = data["equilibrium"]["time_slice"][timeslice]

    dΡ_dρ = eqt["profiles_1d"]["rho_tor"][end]
    
    ρ = eqt["profiles_1d"]["rho_tor"]/dΡ_dρ

    rtype = typeof(ρ[1])
    
    B₀ = data["equilibrium"]["vacuum_toroidal_field"]["b0"][timeslice]

    fsa_R⁻² = FE(ρ, rtype.(eqt["profiles_1d"]["gm1"]))
    F = FE(ρ, rtype.(eqt["profiles_1d"]["f"]))

    # Require dV_dρ=0 on-axis
    tmp = dΡ_dρ .* eqt["profiles_1d"]["dvolume_drho_tor"]
    tmp[1] = 0.0 
    dV_dρ = FE(ρ, tmp)

    ι = FE(ρ, 1.0./eqt["profiles_1d"]["q"])
    JtoR = FE(ρ, eqt["profiles_1d"]["j_tor"] .* eqt["profiles_1d"]["gm9"])

    JBni = nothing
    try
        prof1d = data["core_profiles"]["profiles_1d"][timeslice]
        JBni = FE(rtype.(prof1d["grid"]["rho_tor_norm"]), prof1d["j_non_inductive"] .* B₀)
    catch e
        !(e isa KeyError) && rethrow(e)
    end

    return QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR, JBni=JBni)
end

η_imas(filename::String, timeslice=1) = η_imas(JSON.parsefile(filename), timeslice)

function η_imas(data::Dict, timeslice=1; use_log=true)
    prof1d = data["core_profiles"]["profiles_1d"][timeslice]
    rho = prof1d["grid"]["rho_tor_norm"]
    η = 1.0 ./ prof1d["conductivity_parallel"]
    rtype = typeof(η[1])
    if use_log
        log_η = FE(rtype.(rho), log.(η))
        return x -> exp(log_η(x))
    else
        return FE(rtype.(rho), η)
    end

end

function η_mock(; T0 = 3000.0, Tp = 500.0, Ts = 100.0)
    # Spitzer resistivity in Ωm from NRL (assuming Z=2 and lnΛ=15)
    Te(x) = 0.5*(Tp  + (T0-Tp)*(1.0-x) - Ts)*(1.0 - tanh((x-0.95)/0.025)) + Ts
    return x -> 3.1e-3/Te(x)^1.5
end