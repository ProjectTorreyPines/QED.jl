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
    debug::Bool=false, Nd::Union{Nothing,Integer}=nothing)

    Ic, Mpc, dMpc_dt = build.Ic, build.Mpc, build.dMpc_dt

    T = define_T(QI)
    Y = define_Y(QI, η)

    Δt = tmax / Nt
    inv_Δt = Nt / tmax
    θexp = 1.0 - θimp
    ρ = QI.ρ

    # Factor that translates ι to Ip:  Ip = γ * ι at ρ=1
    γ = dΦ_dρ(QI, 1) * QI.dV_dρ(1) * fsa_∇ρ²_R²(QI, 1)  / ((2π)^2 * μ₀)

    Np = length(ρ)
    Nc = length(build.Ic)
    Ntot = Np + Nc

    A = zeros(Ntot, Ntot)

    ######################
    # Plasma-only matrix
    ######################
    @views Ap = A[1:Np, 1:Np]

    # A = T / Δt - θimp * Y
    Ap .= T .* inv_Δt
    (θimp != 0) && (Ap .-= θimp .* Y)

    # On-axis boundary conditions: ι' = 0
    Ap[1, :] .= 0.0
    Ap[1, 1] = 1.0

    ######################
    # Build-only matrix
    ######################
    @views Ab = A[Np+1:end, Np+1:end]
    Ab .= build_matrix!(build, inv_Δt, θimp)

    #############################
    # Coupling - plasma to build
    #############################

    # Edge loop voltage boundary condition
    # α d(β*ι)/dρ = Vloop at ρ=1
    Ap[end, :] .= 0.0
    Ap[end, end] = αdβ_dρ(1.0, QI, η)
    Ap[end, end-1] = αβ(1.0, QI, η)

    # Forward part of Vloop = sum(Mpc * dIc/dt)
    A[Np, Np+1:end] .= .-Mpc .* inv_Δt

    #############################
    # Coupling - build to plasma
    #############################

    # Forward part of Mpc * dIp_dt
    A[Np+1:end, Np] .= γ .* Mpc .* inv_Δt

    # Implicit part of dMpc_dt * Ip
    (θimp != 0.0) && (A[Np+1:end, Np] .+= θimp .* γ .* dMpc_dt)


    # invert A matrix only once, outside of time stepping loop
    # We could factor and do linear solve each time,
    invA = inv(A)

    if debug
        Nd === nothing && (Nd = Int(floor(Nt^0.75)))
        Ncol = (mod(Nt, Np) == 0) ? Nt ÷ Nd + 2 : Nt ÷ Nd + 3
        ιs = Vector{FE_rep}(undef, Ncol - 2)
        Is = zeros(Nc, Ncol - 2)
        times = zeros(Ncol - 2)
        nd = 0
    end
    Sni = define_Sni(QI, η)

    b = zeros(Ntot)
    @views bp = b[1:Np]
    @views bb = b[Np+1:end]

    c = zeros(Ntot)
    @views cp = c[1:Np]
    @views cb = c[Np+1:end]
    cp .= QI.ι.coeffs
    cb .= build.Ic

    btmp = zeros(Np)

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

        # On-axis boundary condition
        bp[1] = 0.0

        # Edge boundary condition
        bp[end] = -sum(Mpc[k] .* Ic[k] for k in 1:Nc) * inv_Δt

        # Non-inductive source
        Sni !== nothing && (bp .+= Sni)

        ######################
        # Build part
        ######################
        tθ = (n - θexp) * Δt # θ-implicit time
        bb .= build_rhs!(build, tθ, inv_Δt, θexp)
        bb .+= γ .* (inv_Δt .* Mpc .- θexp .* dMpc_dt) .* cp[end]

        mul!(c, invA, b)

        QI.ι.coeffs .= cp
        build.Ic    .= cb

        if debug && ((mod(n, Nd) == 0) || (n == Nt))
            nd += 1
            times[nd] = round(n / inv_Δt; digits=3)
            ιs[nd] = FE_rep(ρ, collect(cp))
            Is[:, nd] .= cb
        end

    end

    ι = FE_rep(ρ, c)
    JtoR = Jt_R(QI; ι=ι)

    if debug
        plot_JtoR_profiles(QI, ιs, times, Ncol)
        display(plot(times, Is', lw=2, palette=Plots.palette(:plasma, Ncol)))
    end
    return QED_state(QI, ι, JtoR)
end