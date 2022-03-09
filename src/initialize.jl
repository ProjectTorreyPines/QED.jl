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

function resample(QI::QED_state, ρ::AbstractVector)

    fsa_R⁻² = resample(QI.fsa_R⁻², ρ)
    F = resample(QI.F, ρ)
    dV_dρ = resample(QI.dV_dρ, ρ)
    ι = resample(QI.ι, ρ)
    JtoR = resample(QI.JtoR, ρ)

    JBni = nothing
    if QI.JBni !== nothing
        JBni = resample(QI.JBni, ρ)
    end

    return QED_state(ρ, QI.dΡ_dρ, QI.B₀, fsa_R⁻², F, dV_dρ, ι, JtoR, JBni=JBni)
end

function optknt(QI::QED_state, N::Integer;
                field=:JtoR,
                ρinit = [0.0, 0.5, 1.0],
                ρtest=range(0.0, 1.0, 10001),
                fit_derivative=false)

    ϵ = eps(typeof(QI.ρ[1]))
    ρ = deepcopy(ρinit)
    if field === :all
        if QI.JBni === nothing
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR]
        else
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR, :JBni]
        end
    elseif field isa Symbol
        fields = [field]
    else
        fields = field
    end

    QK = resample(QI, ρ)

    while length(ρ) < N
        err = zero(ρtest)
        for f in fields
            VI = getproperty(QI, f).(ρtest)
            VK = getproperty(QK, f).(ρtest)
            err .+= abs.(VK - VI) ./ abs.(VI .+ ϵ*maximum(abs.(VI)))
        end
        imax = argmax(err)
        while ρtest[imax] in ρ
            err[imax] *= 0
            imax = argmax(err)
        end

        if fit_derivative
            err = zero(ρtest)
            for f in fields
                VI = D.(Ref(getproperty(QI, f)), ρtest)
                VK = D.(Ref(getproperty(QK, f)), ρtest)
                err .+= abs.(VK - VI) ./ abs.(VI .+ ϵ*maximum(abs.(VI)))
            end
            dmax = argmax(err)
            while ρtest[dmax] in ρ
                err[dmax] *= 0
                dmax = argmax(err)
            end
            dmax != imax && insert!(ρ, searchsortedfirst(ρ, ρtest[dmax]), ρtest[dmax])
        end

        insert!(ρ, searchsortedfirst(ρ, ρtest[imax]), ρtest[imax])
        QK = resample(QI, ρ)
    end

    return QK

end

function optknt2(QI::QED_state, N::Integer;
                 field=:JtoR, ρinit = [0.0, 0.5, 1.0])

    ρ = deepcopy(ρinit)
    if field === :all
        if QI.JBni === nothing
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR]
        else
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR, :JBni]
        end
    elseif field isa Symbol
        fields = [field]
    else
        fields = field
    end

    QK = resample(QI, ρ)

    YIs = [getproperty(QI, f) for f in fields]
    YKs = [deepcopy(getproperty(QK, f)) for f in fields]
    zipYs = zip(YIs, YKs)

    function err(x)
        ϵ = sqrt(eps(typeof(x)))
        cost = 0.0
        for (YI, YK) in zipYs
            vi = YI(x)
            di = D(YI, x)
            vk = YK(x)
            dk = D(YK, x)
            cost += ((vi - vk)/(vi + vk + ϵ))^2
            #cost += ((di - dk)/(di + dk + ϵ))^2
        end
        return cost
    end

    while length(YKs[1].x) < N
        res = optimize(x -> -err(x), 0.0, 1.0)
        ρmax = res.minimizer

        for (YI, YK) in zipYs
            add_point!(YK, ρmax, YI(ρmax), D(YI, ρmax))
        end
    end

    return resample(QI, YKs[1].x)

end

function optknt3(QI::QED_state, N::Integer;
    field=:JtoR, ρinit = [0.0, 0.5, 1.0], opt_on_deriv=false)

    ρ = deepcopy(ρinit)
    if field === :all
        if QI.JBni === nothing
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR]
        else
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR, :JBni]
        end
    elseif field isa Symbol
        fields = [field]
    else
        fields = field
    end

    QK = resample(QI, ρ)

    YIs = [getproperty(QI, f) for f in fields]
    vnorm = [sqrt(sum(getproperty(QI, f).coeffs[2:2:end].^2)) for f in fields]
    dnorm = [sqrt(sum(getproperty(QI, f).coeffs[1:2:end].^2)) for f in fields]
    YKs = [deepcopy(getproperty(QK, f)) for f in fields]
    zipYs = zip(YIs, YKs, vnorm, dnorm)

    function err(x)
        ϵ = sqrt(eps(typeof(x)))
        cost = 0.0
        for (YI, YK, vn, dn) in zipYs
            vi = YI(x)
            di = D(YI, x)
            vk = YK(x)
            dk = D(YK, x)
            cost += ((vi - vk)/vn)^2
            opt_on_deriv && (cost += ((di - dk)/dn)^2)
        end
        return cost
    end

    errs = zeros(length(ρ)-1)
    ρs = zeros(length(ρ)-1)
    first = true
    imax = 1

    while length(ρ) < N
        M = length(ρ)

        Is = nothing
        first ? Is = range(1,M-1) : Is = [imax, imax+1]

        for i in Is
            res = Optim.optimize(x -> -err(x), ρ[i], ρ[i+1])
            errs[i] = -res.minimum
            ρs[i] = res.minimizer
        end
        #println(ρs)
        #println(errs)
        #println()
        first = false

        imax = argmax(errs)
        ρmax = ρs[imax]
        errs[imax] = 0.0
        ρs[imax] = 0.0
        insert!(errs, imax, 0.0)
        insert!(ρs, imax, 0.0)


        jmax = searchsortedfirst(ρ, ρmax)
        insert!(ρ, jmax, ρmax)

        for (YI, YK, _, _) in zipYs
            add_point!(YK, ρmax, YI(ρmax), D(YI, ρmax))
        end
    end

    return resample(QI, ρ)

end

function optknt4(QI::QED_state, N::Integer;
    field=:JtoR, ρinit = [0.0, 0.5, 1.0], opt_on_deriv=false)

    ρ = deepcopy(ρinit)
    if field === :all
        if QI.JBni === nothing
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR]
        else
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR, :JBni]
        end
    elseif field isa Symbol
        fields = [field]
    else
        fields = field
    end

    QK = resample(QI, ρ)

    YIs = [getproperty(QI, f) for f in fields]
    vnorm = [sqrt(sum(getproperty(QI, f).coeffs[2:2:end].^2)) for f in fields]
    dnorm = [sqrt(sum(getproperty(QI, f).coeffs[1:2:end].^2)) for f in fields]
    YKs = [deepcopy(getproperty(QK, f)) for f in fields]
    zipYs = zip(YIs, YKs, vnorm, dnorm)

    function err(xv)
        x = xv[1]
        #ϵ = sqrt(eps(typeof(x)))
        cost = 0.0
        for (YI, YK, vn, dn) in zipYs
            vi = YI(x)
            di = D(YI, x)
            vk = YK(x)
            dk = D(YK, x)
            cost += ((vi - vk)/vn)^2
            opt_on_deriv && (cost += ((di - dk)/dn)^2)
        end
        return cost
    end

    errs = zeros(length(ρ)-1)
    ρs = zeros(length(ρ)-1)
    first = true
    imax = 1

    algo = Optim.Fminbox(Optim.BFGS())

    while length(ρ) < N
        M = length(ρ)

        Is = nothing
        first ? Is = range(1,M-1) : Is = [imax, imax+1]

        for i in Is
            ρmin = [ρ[i]]
            ρmax = [ρ[i+1]]
            ρ0 = 0.5 .* (ρmin .+ ρmax)
            res = Optim.optimize(x -> -err(x), ρmin, ρmax, ρ0, algo)
            errs[i] = -res.minimum[1]
            ρs[i] = res.minimizer[1]
        end
        #println(ρs)
        #println(errs)
        #println()
        first = false

        imax = argmax(errs)
        ρmax = ρs[imax]
        errs[imax] = 0.0
        ρs[imax] = 0.0
        insert!(errs, imax, 0.0)
        insert!(ρs, imax, 0.0)


        jmax = searchsortedfirst(ρ, ρmax)
        insert!(ρ, jmax, ρmax)

        for (YI, YK, _, _) in zipYs
            add_point!(YK, ρmax, YI(ρmax), D(YI, ρmax))
        end
    end

    return resample(QI, ρ)

end

function optknt5(QI::QED_state, N::Integer;
                 field=:JtoR)

    if field === :all
        if QI.JBni === nothing
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR]
        else
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR, :JBni]
        end
    elseif field isa Symbol
        fields = [field]
    else
        fields = field
    end

    function rho(dρ)
        ρ = zeros(length(dρ)+1)
        for i in 1:(N-1)
            ρ[i+1] = ρ[i] + dρ[i]
        end
        ρ ./= ρ[end]
        return ρ
    end

    vnorm = [sqrt(sum(getproperty(QI, f).coeffs[2:2:end].^2)) for f in fields]
    dnorm = [sqrt(sum(getproperty(QI, f).coeffs[1:2:end].^2)) for f in fields]
    fvd = zip(fields, vnorm, dnorm)

    function err(dρ)
        ρ = rho(dρ)
        cost = 0.0
        for (f, vn, dn) in fvd
            YI = getproperty(QI, f)
            vi = YI.coeffs[2:2:end]
            di = YI.coeffs[1:2:end]

            YK = resample(YI, ρ)
            vk = YK.(YI.x)
            dk = D.(Ref(YK), YI.x)

            cost += sum(((vi - vk)/vn).^2)
            #cost += ((di - dk)/dn)^2
        end
        return cost
    end

    algo = Optim.Fminbox(Optim.BFGS())
    dρ0 = ones(N-1) ./ (N-1)
    lower = zeros(length(dρ0))
    upper = ones(length(dρ0)) ./ sqrt(N-1.0)
    res = Optim.optimize(err, lower, upper, dρ0, algo)

    return resample(QI, rho(res.minimizer))

end

function autoknt(QI::QED_state, N::Integer;
                 field=:JtoR)

    if field === :all
        if QI.JBni === nothing
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR]
        else
            fields = [:fsa_R⁻², :F, :dV_dρ, :ι, :JtoR, :JBni]
        end
    elseif field isa Symbol
        fields = [field]
    else
        fields = field
    end

    M = 2*length(QI.ρ) - 1
    ρ = zeros(M)
    ρ[1:2:end] = QI.ρ
    ρ[2:2:end] = 0.5*(QI.ρ[1:(end-1)] + QI.ρ[2:end])
    ϵ = eps(typeof(ρ[1]))

    y = range(0.0, 1.0, length=N)
    dy = zero(ρ)
    for f in fields
        vf = getproperty(QI, f)

        df = abs.(D.(Ref(vf), ρ) ./ (vf.(ρ) .+ ϵ))
        #df ./= maximum(df)
        #df .+= 0.01

        #d2f = abs.(fit_derivative(ρ, D.(Ref(vf),ρ)))
        #d2f ./= maximum(d2f)
        dy .+= df #+ d2f
    end
    FE_d = FE(ρ, dy)
    Y = I.(Ref(FE_d),ρ)
    Y ./= Y[end]
    Y[1] = 0.0

    y2ρ = FE(Y, ρ)
    ρnew = 0.5 .* (y2ρ.(y) + y)
    ρnew = y2ρ.(y)

    return resample(QI, ρnew)

end