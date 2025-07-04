mutable struct QED_state{U<:AbstractVector{<:Real},T<:Real,S<:FE_rep,B<:Union{Nothing, QED_build}}
    ρ::U
    dΡ_dρ::T
    B₀::T
    fsa_R⁻²::S
    F::S
    dV_dρ::S
    ι::S
    JtoR::S
    χ::S
    JBni::Union{Nothing,S}
    _ι_eq::S
    build::B
end

QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR, χ, JBni, _ι_eq) = QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR, χ, JBni, _ι_eq, nothing)

@inline dΦ_dρ(QI::QED_state, x::Real) = 2π * QI.B₀ * QI.dΡ_dρ^2 * x

@inline function fsa_∇ρ²_R²(QI::QED_state, r::Real; ε=1e-3)
    r == 0 && return 2 * fsa_∇ρ²_R²(QI, ε) - fsa_∇ρ²_R²(QI, 2ε) # Linearly extrapolate to axis
    return QI.χ(r) / (QI.dV_dρ(r) * QI._ι_eq(r) * dΦ_dρ(QI, r))
end

@inline function D_fsa_∇ρ²_R²(QI::QED_state, x::Real)
    return ForwardDiff.derivative(r -> fsa_∇ρ²_R²(QI, r), x)
end

function QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR; JBni=nothing, Ip0=nothing,
    x=1.0 .- (1.0 .- range(0, 1; length=length(ρ))) .^ 2)

    # If ξ = 2π*μ₀ * dV_dρ * <Jt/R> and χ = dV_dρ * dΨ_dρ * <|∇ρ|²/R²>
    # where dΨ_dρ = ι * dΦ_dρ
    # Then dχ/dρ = ξ
    # Solve for χ by integrating ξ, noting that χ=0 at ρ=0
    ξ = FE(x, 2π * μ₀ * dV_dρ.(x) .* JtoR.(x))
    C = zeros(2 * length(x))
    C[2:2:end] .= I.(Ref(ξ), x) # value of χ is integral of ξ
    C[1:2:end] .= ξ.(x)         # derivative of χ is value of ξ
    χ = FE_rep(x, C)

    Q = QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR, χ, JBni, ι)

    # Scale the last value of χ to match the desired Ip0
    # Can modify the value in C directly
    if Ip0 !== nothing
        fac = Ip0 / Ip(Q)
        C[end] *= fac
    end

    return Q
end

function QED_state(QI::QED_state, ι, JtoR)
    ι isa AbstractVector && (ι = FE(QI.ρ, ι))
    JtoR isa AbstractVector && (JtoR = FE(QI.ρ, JtoR))

    return QED_state(QI.ρ, QI.dΡ_dρ, QI.B₀, QI.fsa_R⁻², QI.F, QI.dV_dρ, ι, JtoR, QI.χ, QI.JBni, QI._ι_eq)
end

function QED_state(QI::QED_state; JBni=nothing)
    if JBni isa AbstractVector
        JBni = FE(QI.ρ, JBni)
    elseif JBni !== nothing
        JBni = FE(QI.ρ, JBni.(QI.ρ))
    end
    return QED_state(QI.ρ, QI.dΡ_dρ, QI.B₀, QI.fsa_R⁻², QI.F, QI.dV_dρ, QI.ι, QI.JtoR, QI.χ, JBni, QI._ι_eq)
end

"""
    from_imas(filename::String, timeslice=1)

Initialize the QED data structure from a JSON `filename` in IMAS format at requested `timeslice`
"""
from_imas(filename::String, timeslice=1) = from_imas(JSON.parsefile(filename), timeslice)

"""
    from_imas(data::Dict, timeslice=1)

Initialize the QED data structure from a dictionary `data` in IMAS format at requested `timeslice`
"""
function from_imas(data::Dict, timeslice=1)

    eqt = data["equilibrium"]["time_slice"][timeslice]
    rho_tor = Float64.(eqt["profiles_1d"]["rho_tor"])
    B₀ =  Float64.(data["equilibrium"]["vacuum_toroidal_field"]["b0"][timeslice])
    gm1 =  Float64.(eqt["profiles_1d"]["gm1"])
    f =  Float64.(eqt["profiles_1d"]["f"])
    dvolume_drho_tor =  Float64.(eqt["profiles_1d"]["dvolume_drho_tor"])
    q =  Float64.(eqt["profiles_1d"]["q"])
    j_tor =  Float64.(eqt["profiles_1d"]["j_tor"])
    gm9 =  Float64.(eqt["profiles_1d"]["gm9"])

    ρ_j_non_inductive = nothing
    try
        prof1d = data["core_profiles"]["profiles_1d"][timeslice]
        ρ_j_non_inductive = ( Float64.(prof1d["grid"]["rho_tor_norm"]),  Float64.(prof1d["j_non_inductive"]))
    catch e
        !(e isa KeyError) && rethrow(e)
    end

    return initialize(rho_tor, B₀, gm1, f, dvolume_drho_tor, q, j_tor, gm9; ρ_j_non_inductive)
end

"""
    initialize(rho_tor::AbstractVector{<:Real},
               B₀::Real,
               gm1::AbstractVector{<:Real},
               f::AbstractVector{<:Real},
               dvolume_drho_tor::AbstractVector{<:Real},
               q::AbstractVector{<:Real},
               j_tor::AbstractVector{<:Real},
               gm9::AbstractVector{<:Real};
               ρ_j_non_inductive::Union{Nothing,Tuple{<:AbstractVector{<:Real},<:AbstractVector{<:Real}}}=nothing,
               ρ_grid::Union{Nothing,AbstractVector{<:Real}}=nothing)

Initialize the QED data structure with data typically taken from an IMAS-like data structure, e.g.,
```
eqt = data["equilibrium"]["time_slice"][timeslice]
rho_tor = eqt["profiles_1d"]["rho_tor"]
B₀ = data["equilibrium"]["vacuum_toroidal_field"]["b0"][timeslice]
gm1 = eqt["profiles_1d"]["gm1"]
f = eqt["profiles_1d"]["f"]
dvolume_drho_tor = eqt["profiles_1d"]["dvolume_drho_tor"]
q = eqt["profiles_1d"]["q"]
j_tor = eqt["profiles_1d"]["j_tor"]
gm9 = eqt["profiles_1d"]["gm9"]
```
Keyword arguments:
`ρ_j_non_inductive` - a tuple giving a grid and the noninductive current
                    e.g., `ρ_j_non_inductive = (prof1d["grid"]["rho_tor_norm"], prof1d["j_non_inductive"])`
`ρ_grid` - The grid for QED to operate on. By default, it uses normalized `rho_tor`
"""
function initialize(rho_tor::AbstractVector{<:Real},
                    B₀::Real,
                    gm1::AbstractVector{<:Real},
                    f::AbstractVector{<:Real},
                    dvolume_drho_tor::AbstractVector{<:Real},
                    q::AbstractVector{<:Real},
                    j_tor::AbstractVector{<:Real},
                    gm9::AbstractVector{<:Real};
                    ρ_j_non_inductive::Union{Nothing,Tuple{<:AbstractVector{<:Real},<:AbstractVector{<:Real}}}=nothing,
                    ρ_grid::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    Ip0::Union{Nothing,Real}=nothing)

    dΡ_dρ = rho_tor[end]

    ρ = rho_tor / dΡ_dρ

    rtype = typeof(ρ[1])

    fsa_R⁻² = FE(ρ, rtype.(gm1))
    F = FE(ρ, rtype.(f))

    # Require dV_dρ=0 on-axis
    tmp = dΡ_dρ .* dvolume_drho_tor
    tmp[1] = 0.0
    dV_dρ = FE(ρ, tmp)

    JtoR = FE(ρ, j_tor .* gm9)

    if ρ_j_non_inductive === nothing
        JBni = nothing
    else
        rho_tor_norm, j_non_inductive = ρ_j_non_inductive
        JBni = FE(rtype.(rho_tor_norm), j_non_inductive .* B₀)
    end

    if ρ_grid !== nothing
        ι = FE(ρ_grid, (ρ, 1.0 ./ q))
        return QED_state(ρ_grid, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR; JBni, Ip0)
    else
        ι = FE(ρ, 1.0 ./ q)
        return QED_state(ρ, dΡ_dρ, B₀, fsa_R⁻², F, dV_dρ, ι, JtoR; JBni, Ip0)
    end
end

"""
    η_imas(filename::String, timeslice::Integer=1; use_log::Bool=true)

Return an interpolation of the resistivity from an IMAS-like JSON file `filename` at `timeslice`
`use_log=true` (default) interpolates on the log of the resistivity
"""
η_imas(filename::String, timeslice::Integer=1; use_log::Bool=true) = η_imas(JSON.parsefile(filename), timeslice; use_log)

"""
    η_imas(data::Dict, timeslice::Integer=1; use_log::Bool=true)

Return an interpolation of the resistivity from an IMAS-like dictionary `data` at `timeslice`
`use_log=true` (default) interpolates on the log of the resistivity
"""
function η_imas(data::Dict, timeslice::Integer=1; use_log::Bool=true)
    prof1d = data["core_profiles"]["profiles_1d"][timeslice]
    rho = Float64.(prof1d["grid"]["rho_tor_norm"])
    η = 1.0 ./ Float64.(prof1d["conductivity_parallel"])
    return η_FE(rho, η; use_log)
end

"""
    η_FE(rho::AbstractVector{<:Real}, η::AbstractVector{<:Real}; use_log::Bool=true)

Return a cubic-Hermite finite-element interpolation of the resistivity `η` on grid `rho`
`use_log=true` (default) interpolates on the log of the resistivity
"""
function η_FE(rho::AbstractVector{S}, η::AbstractVector{T}; use_log::Bool=true) where {S<:Real,T<:Real}
    rhot = S.(rho)
    return η_FE(rhot, η; use_log)
end

function η_FE(rho::AbstractVector{T}, η::AbstractVector{T}; use_log::Bool=true) where {T<:Real}
    if use_log
        log_η = FE(rho, log.(η))
        return x -> exp(log_η(x))
    else
        return FE(rho, η)
    end
end

"""
    η_mock(; T0::Real=3000.0, Tp::Real=500.0, Ts::Real=100.0)

Return a function of a mock resistivity profile with temperatures `T0` (core), `Tp` (pedestal), and `Ts` (separatrix)
"""
function η_mock(; T0::Real=3000.0, Tp::Real=500.0, Ts::Real=100.0)
    # Spitzer resistivity in Ωm from NRL (assuming Z=2 and lnΛ=15)
    Te(x) = 0.5 * (Tp + (T0 - Tp) * (1.0 - x) - Ts) * (1.0 - tanh((x - 0.95) / 0.025)) + Ts
    return x -> 3.1e-3 / Te(x)^1.5
end


# dummy for IMAS_Ext
function η_JBni_sawteeth end