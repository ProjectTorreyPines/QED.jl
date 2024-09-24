function build_matrix!(build::QED_build, inv_Δt::Real, θimp::Real)
    Rc, Mcc, A = build.Rc, build.Mcc, build._A
    A .= inv_Δt .* Mcc
    (θimp != 0.0) && (A.+= θimp .* Diagonal(Rc))
    return A
end

function build_rhs!(build::QED_build, t, inv_Δt::Real, θexp::Real)
    Ic, Vc, Rc, Mcc, b = build.Ic, build.Vc, build.Rc, build.Mcc, build._b

    # explicit inductive and resistive terms
    mul!(b, Mcc, Ic)
    b .*= inv_Δt
    (θexp != 0.0) && (b .-= θexp .* Rc .* Ic)

    # source voltage at requested time
    update_voltages!(build, t)
    b += Vc
    return b
end


# This does evolution of the build currents, assuming no plasma
function evolve!(build::QED_build, tmax::Real, Nt::Integer; θimp::Real=0.5)
    Ic = build.Ic
    Nc = length(Ic)
    Is = zeros(Nc, Nt+1)
    Is[:,1] .= Ic

    Δt = tmax / Nt
    inv_Δt = 1.0 / Δt
    θexp = 1.0 - θimp

    # A matrix is implicit inductive and resistive term
    # unchanging so we can store inverse
    # BCL 9/12/24: This should be factorization object and use a linear solver
    A = build_matrix!(build, inv_Δt, θimp)
    invA = inv(A)

    for n in 1:Nt
        tθ = (n - θexp) * Δt # θ-implicit time
        b = build_rhs!(build, tθ, inv_Δt, θexp)
        mul!(Ic, invA, b)
        Is[:,n+1] .= Ic
    end
    return Is
end

"""
    evolve(QI::QED_state, η, build::QED_build, tmax::Real, Nt::Integer;
            θimp::Real=0.5,
            debug::Bool=false, Np::Union{Nothing,Integer}=nothing)

Evolve rotational transform in QED_state `QI` with resistivity `η` and
surrounding structures `build` for `tmax` seconds in `Nt`
Keyword arguments
    `θimp`  - 0 is fully explicit, 1 is fully implicity, 0.5 (default) gives second-order time accuracy
    `debug` - plot intermediate rotational-transform profiles
    `Np`    - if `debug` is true, number of time steps between plots of rotational transform
"""
function evolve(QI::QED_state, η, build::QED_build, tmax::Real, Nt::Integer;
    θimp::Real=0.5,
    debug::Bool=false, Np::Union{Nothing,Integer}=nothing)

    Ic, Vni, Rp, Lp, Mpc, dMpc_dt = build.Ic, build.Vni, build.Rp, build.Lp, build.Mpc, build.dMpc_dt

    T = define_T(QI)
    Y = define_Y(QI, η)

    Δt = tmax / Nt
    inv_Δt = Nt / tmax
    θexp = 1.0 - θimp
    ρ = QI.ρ

    # Factor that translates ι to Ip:  Ip = γ * ι at ρ=1
    γ = dΦ_dρ(QI, 1) * QI.dV_dρ(1) * fsa_∇ρ²_R²(QI, 1)  / ((2π)^2 * μ₀)

    Ni = 2length(ρ)
    Nc = length(build.Ic)
    Ntot = Ni + Nc

    A = zeros(Ntot, Ntot)

    ######################
    # Plasma-only matrix
    ######################
    @views Ap = A[1:Ni, 1:Ni]

    # A = T / Δt - θimp * Y
    Ap .= T .* inv_Δt
    (θimp != 0) && (Ap .-= θimp .* Y)

    # On-axis boundary conditions: ι' = 0
    Ap[1, :] .= 0.0
    Ap[1, 1] = 1.0

    Ap[end, :] .= 0.0
    Ap[end, end] = γ * (Lp * inv_Δt + θimp * Rp)

    ######################
    # Build-only matrix
    ######################
    @views Ab = A[Ni+1:end, Ni+1:end]
    Ab .= build_matrix!(build, inv_Δt, θimp)


    #############################
    # Coupling - plasma to build
    #############################
    p2b = true

    # Forward part of Vloop = sum(Mpc * dIc/dt)
    A[Ni, Ni+1:end] .= p2b ? Mpc .* inv_Δt : 0.0

    #############################
    # Coupling - build to plasma
    #############################
    b2p = true

    if b2p
        # Forward part of Mpc * dIp_dt
        A[Ni+1:end, Ni] .= γ .* Mpc .* inv_Δt

        # Implicit part of dMpc_dt * Ip
        (θimp != 0.0) && (A[Ni+1:end, Ni] .+= θimp .* γ .* dMpc_dt)
    else
        A[Ni+1:end, Ni] .= 0.0
    end

    # invert A matrix only once, outside of time stepping loop
    # We could factor and do linear solve each time,
    invA = inv(A)

    if debug
        Np === nothing && (Np = Int(floor(Nt^0.75)))
        Ncol = (mod(Nt, Np) == 0) ? Nt ÷ Np + 2 : Nt ÷ Np + 3
        ιs = Vector{FE_rep}(undef, Ncol - 2)
        Is = zeros(Nc, Ncol - 2)
        times = zeros(Ncol - 2)
        np = 0
    end
    Sni = define_Sni(QI, η)

    b = zeros(Ntot)
    @views bp = b[1:Ni]
    @views bb = b[Ni+1:end]

    c = zeros(Ntot)
    @views cp = c[1:Ni]
    @views cb = c[Ni+1:end]
    cp .= QI.ι.coeffs
    cb .= build.Ic

    btmp = zeros(Ni)

    # We advance c, the coefficients of ι,
    #   but we never actually need ι until the end (except for debugging)
    for n in 1:Nt

        ######################
        # Plasma part
        ######################
        mul!(bp, T, cp)
        rmul!(bp, inv_Δt)

        if θexp != 0.0
            mul!(btmp, Y, cp)
            rmul!(btmp, θexp)
            bp .+= btmp
        end

        # Edge boundary condition
        bp[end] = Vni + γ * (Lp * inv_Δt - θexp * Rp) * cp[end]
        bp[end] += p2b ? sum(Mpc[k] * Ic[k] for k in 1:Nc) * inv_Δt : 0.0

        # Non-inductive source
        Sni !== nothing && (bp .+= Sni)

        # On-axis boundary condition
        bp[1] = 0.0

        ######################
        # Build part
        ######################
        tθ = (n - θexp) * Δt # θ-implicit time
        bb .= build_rhs!(build, tθ, inv_Δt, θexp)
        if b2p
            bb .+= γ .* (inv_Δt .* Mpc .- θexp .* dMpc_dt) .* cp[end]
        end

        mul!(c, invA, b)

        build.Ic .= cb

        if debug && ((mod(n, Np) == 0) || (n == Nt))
            np += 1
            times[np] = round(n / inv_Δt; digits=3)
            ιs[np] = FE_rep(ρ, collect(cp))
            Is[:, np] .= cb
        end

    end

    ι = FE_rep(ρ, collect(cp))
    JtoR = Jt_R(QI; ι=ι)

    if debug
        plot_JtoR_profiles(QI, ιs, times, Ncol)
        display(Plots.plot(times, Is', lw=2, palette=Plots.palette(:plasma, Ncol), legend=nothing))
    end
    return QED_state(QI, ι, JtoR)
end