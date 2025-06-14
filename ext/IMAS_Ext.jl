module TEQUILA_Ext

import QED, IMAS

"""
    initialize(dd::IMAS.dd, qmin_desired::Union{Nothing, Real}=nothing; uniform_rho::Int, j_tor_from::Symbol=:core_profiles, ip_from::Union{Symbol,Real}=j_tor_from) where {D<:Real,P<:Real}

Setup QED from data in IMAS `dd`
"""
function QED.initialize(dd::IMAS.dd, qmin_desired::Union{Nothing, Real}=nothing; uniform_rho::Int, j_tor_from::Symbol=:core_profiles, ip_from::Union{Symbol,Real}=j_tor_from)

    eqt = dd.equilibrium.time_slice[]
    cp1d = dd.core_profiles.profiles_1d[]
    B0 = eqt.global_quantities.vacuum_toroidal_field.b0

    rho_tor = eqt.profiles_1d.rho_tor
    gm1 = eqt.profiles_1d.gm1
    f = eqt.profiles_1d.f
    dvolume_drho_tor = eqt.profiles_1d.dvolume_drho_tor
    q = eqt.profiles_1d.q
    gm9 = eqt.profiles_1d.gm9

    # DO NOT use the equilibrium j_tor, since it's quality depends on the quality/resolution of the equilibrium solver
    # better to use the j_tor from core_profiles, which is the same quantity that is input in the equilibrium solver
    if j_tor_from === :equilibrium
        j_tor = eqt.profiles_1d.j_tor
    elseif j_tor_from === :core_profiles
        j_tor = IMAS.interp1d(cp1d.grid.rho_tor_norm, cp1d.j_tor, :cubic).(IMAS.norm01(rho_tor))
    else
        error("j_tor_from must be :equilibrium or :core_profiles")
    end

    if ip_from === :equilibrium
        Ip0 = IMAS.get_from(dd, Val{:ip}, :equilibrium)
    elseif ip_from === :core_profiles
        Ip0 = IMAS.get_from(dd, Val{:ip}, :core_profiles)
    elseif typeof(ip_from) <: Real
        Ip0 = ip_from
    else
        error("ip_from must be :equilibrium, :core_profiles, or a real number")
    end

    if ismissing(cp1d, :j_non_inductive)
        ρ_j_non_inductive = nothing
    elseif qmin_desired === nothing
        ρ_j_non_inductive = (cp1d.grid.rho_tor_norm, cp1d.j_non_inductive)
    else
        i_qdes = findlast(abs.(eqt.profiles_1d.q) .< qmin_desired)
        if i_qdes === nothing
            rho_qdes = -1.0
        else
            rho_qdes = eqt.profiles_1d.rho_tor_norm[i_qdes]
        end
        _, j_non_inductive = η_JBni_sawteeth(cp1d, cp1d.j_non_inductive, rho_qdes)
        ρ_j_non_inductive = (cp1d.grid.rho_tor_norm, j_non_inductive)
    end

    ρ_grid = collect(range(0.0, 1.0, uniform_rho))

    return QED.initialize(rho_tor, B0, gm1, f, dvolume_drho_tor, q, j_tor, gm9; ρ_j_non_inductive, ρ_grid, Ip0)
end

"""
    η_imas(cp1d::IMAS.core_profiles__profiles_1d; use_log::Bool=true)

returns the resistivity profile as a function of rho_tor_norm

    - `use_log=true`: Cubic finite element interpolation on a log scale

    - `use_log=false` Cubic finite element interpolation on a linear scale
"""
function η_imas(cp1d::IMAS.core_profiles__profiles_1d; use_log::Bool=true)
    rho = cp1d.grid.rho_tor_norm
    η = 1.0 ./ cp1d.conductivity_parallel
    return QED.η_FE(rho, η; use_log)
end

"""
    η_imas(dd::IMAS.dd; use_log::Bool=true)

returns the resistivity profile as a function of rho_tor_norm from core_profiles

    - `use_log=true`: Cubic finite element interpolation on a log scale

    - `use_log=false` Cubic finite element interpolation on a linear scale
"""
η_imas(dd::IMAS.dd; use_log::Bool=true) = η_imas(dd.core_profiles.profiles_1d[]; use_log)


end