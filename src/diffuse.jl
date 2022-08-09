# Matrix for the time-derivative term (without Δt)
# T_mk = (dΦ_dρ * ν_m * ν_k)
function define_T(QI::QED_state, order::Integer = 5)

    ρ = QI.ρ
    dΦ_dρ = QI.dΦ_dρ
    
    N = length(ρ)

    Ts = zeros(2N, 7) # Hermite cubics give septadiagonal matrix

    Threads.@threads for m in 1:N
        # First odd with all nearest neighbors
        Ts[2m-1, 1] = 0.0
        if m > 1
            Ts[2m-1, 2] = inner_product(dΦ_dρ, νo, m, νo, m-1, ρ, order)
            Ts[2m-1, 3] = inner_product(dΦ_dρ, νo, m, νe, m-1, ρ, order)
        end
        Ts[2m-1, 4] = inner_product(dΦ_dρ, νo, m, νo, m, ρ, order)
        Ts[2m-1, 5] = inner_product(dΦ_dρ, νo, m, νe, m, ρ, order)
        if m < N
            Ts[2m-1, 6] = inner_product(dΦ_dρ, νo, m, νo, m+1, ρ, order)
            Ts[2m-1, 7] = inner_product(dΦ_dρ, νo, m, νe, m+1, ρ, order)
        end

        # Then even with all nearest neighbors
        if m > 1
            Ts[2m, 1] = inner_product(dΦ_dρ, νe, m, νo, m-1, ρ, order)
            Ts[2m, 2] = inner_product(dΦ_dρ, νe, m, νe, m-1, ρ, order)
        end
        Ts[2m, 3] = inner_product(dΦ_dρ, νe, m, νo, m, ρ, order)
        Ts[2m, 4] = inner_product(dΦ_dρ, νe, m, νe, m, ρ, order)
        if m < N
            Ts[2m, 5] = inner_product(dΦ_dρ, νe, m, νo, m+1, ρ, order)
            Ts[2m, 6] = inner_product(dΦ_dρ, νe, m, νe, m+1, ρ, order)
        end
        Ts[2m, 7] = 0.0
    end

    return BandedMatrix(-3 => Ts[4:end,1], -2 => Ts[3:end,2],   -1 => Ts[2:end,3], 0 => Ts[:,4],
                         1 => Ts[1:end-1,5],  2 => Ts[1:end-2,6], 3 => Ts[1:end-3,7])

end

function αβ(x::Real, QI::QED_state, η)
    return η(x) * QI.dΦ_dρ(x) * QI.fsa_∇ρ²_R²(x) / (μ₀ * QI.fsa_R⁻²(x))
end

function αdβ_dρ(x::Real, QI::QED_state, η)

    if x != 0
        γ = x * (D(QI.dV_dρ, x) / QI.dV_dρ(x) - D(QI.F, x) / QI.F(x))
    else
        γ = 1.0
    end
    abp = (1.0 + γ) * QI.fsa_∇ρ²_R²(x) + x * QI.D_fsa_∇ρ²_R²(x)
    return 2π * QI.B₀ * QI.dΡ_dρ^2 * η(x) * abp / (μ₀ * QI.fsa_R⁻²(x))
end

# Matrix for the diffusion term
# Y_mk = (ν_m * d/dρ[α * d/dρ(β * ν_k)])
# This gets integrated by parts to avoid second derivatives of finite elements
function define_Y(QI::QED_state, η, order::Integer = 5)

    ρ = QI.ρ
    N = length(ρ)

    # transform to single argument functions for inner_product
    ab(x) = αβ(x, QI, η)
    adb_dρ(x) = αdβ_dρ(x, QI, η)
    
    # To speed up, if accurate
    #x = 1.0 .- (1.0 .- range(0.0, 1.0,  2N)).^2
    #ab = FE(x, αβ.(x, Ref(QI), Ref(η)))
    #adb_dρ = FE(x, αdβ_dρ.(x, Ref(QI), Ref(η)))    

    Ys = zeros(2N, 7) # Hermite cubics give septadiagonal matrix
    
    Threads.@threads for m in 1:N

        # First odd with all nearest neighbors
        Ys[2m-1, 1] = 0.0
        if m > 1
            Ys[2m-1, 2]  = -inner_product(D_νo, m, ab, D_νo, adb_dρ, νo, m-1, ρ, order)
            Ys[2m-1, 3]  = -inner_product(D_νo, m, ab, D_νe, adb_dρ, νe, m-1, ρ, order)
        end
        
        Ys[2m-1, 4]  = -inner_product(D_νo, m, ab, D_νo, adb_dρ, νo, m, ρ, order)
        Ys[2m-1, 5]  = -inner_product(D_νo, m, ab, D_νe, adb_dρ, νe, m, ρ, order)

        if m < N
            Ys[2m-1, 6]  = -inner_product(D_νo, m, ab, D_νo, adb_dρ, νo, m+1, ρ, order)
            Ys[2m-1, 7]  = -inner_product(D_νo, m, ab, D_νe, adb_dρ, νe, m+1, ρ, order)
        end

        # Then even with all nearest neighbors
        if m > 1
            Ys[2m, 1]  = -inner_product(D_νe, m, ab, D_νo, adb_dρ, νo, m-1, ρ, order)
            Ys[2m, 2]  = -inner_product(D_νe, m, ab, D_νe, adb_dρ, νe, m-1, ρ, order)
        end
        
        Ys[2m, 3]  = -inner_product(D_νe, m, ab, D_νo, adb_dρ, νo, m, ρ, order)
        Ys[2m, 4]  = -inner_product(D_νe, m, ab, D_νe, adb_dρ, νe, m, ρ, order)

        if m < N
            Ys[2m, 5]  = -inner_product(D_νe, m, ab, D_νo, adb_dρ, νo, m+1, ρ, order)
            Ys[2m, 6]  = -inner_product(D_νe, m, ab, D_νe, adb_dρ, νe, m+1, ρ, order)
        end
        Ys[2m, 7] = 0.0
    end
    
    # Y matrix has an integration by parts, so we need the boundary terms

    # Term at ρ=1 goes in the 2N row (only element non-zero at ρ=1)
    Ys[2N, 4] += adb_dρ(1.0) # νe(1.0, N, ρ) is 1.0
    Ys[2N, 3] += ab(1.0) * D_νo(1.0, N, ρ)
    
    # Term at ρ=0 uses fact β(0)=0 and ρ * (F/V') * d(V'/F)/dρ = 1.0
    Ys[2, 4] -= adb_dρ(0.0)
    
    return BandedMatrix(-3 => Ys[4:end,1],   -2 => Ys[3:end,2],   -1 => Ys[2:end,3], 0 => Ys[:,4],
                         1 => Ys[1:end-1,5],  2 => Ys[1:end-2,6],  3 => Ys[1:end-3,7])

end

function define_Sni(QI::QED_state, η, order::Integer = 5)

    QI.JBni === nothing && return nothing

    ρ = QI.ρ
    N = length(ρ)

    V(x) = Vni(QI, η, x)

    Sni = zeros(2N)
    for m in 1:N
        Sni[2m-1] = inner_product(V, D_νo, m, ρ, order)
        Sni[2m]   = inner_product(V, D_νe, m, ρ, order)
    end

    # Boundary terms from integration by parts
    # Only need first and last even, since only ones nonzero on boundary
    Sni[2N] -= Vni(QI, η, 1.0)# * νe(1.0, N, ρ)
    Sni[2]  += Vni(QI, η, 0.0)# * νe(0.0, 1, ρ)

    return Sni
end

function diffuse(QI::QED_state, η, tmax::Real, Nt::Integer;
                 θimp = 0.5,
                 T = nothing, Y = nothing,
                 Vedge = nothing, Ip = nothing,
                 debug = false, Np = nothing)

    T === nothing && (T = define_T(QI))
    Y === nothing && (Y = define_Y(QI, η))

    Δt = tmax / Nt
    ρ = QI.ρ

    A = T ./ Δt 
    if θimp != 0
        A -= θimp .* Y
    end

    # On-axis oundary conditions
    # ι' = 0
    A[1, :] .= 0.0
    A[1, 1] = 1.0

    # Edge boundary condtions
    A[end, :] .= 0.0
    if Vedge !== nothing
        # Constant loop voltage
        # α d(β*ι)/dρ = Vedge at ρ=1
        A[end, end] = αdβ_dρ(1.0, QI, η)
        A[end, end-1] = αβ(1.0, QI, η)
    else
        # Constant ι (i.e., current)
        A[end, end] = 1.0
    end

    if debug
        Np === nothing && (Np = Int(floor(Nt^0.75)))
        mod(Nt, Np) == 0 ? Ncol = Nt÷Np + 2 : Ncol = Nt÷Np + 3
        ιs = Vector{FE_rep}(undef, Ncol-2)
        times = zeros(Ncol-2)
        np = 0
    end

    ι = deepcopy(QI.ι)

    Sni = define_Sni(QI, η)

    # invert A matrix only once, outside of time stepping loop
    invA = inv(A)

    for n in 1:Nt
    
        if θimp != 1.0
            b = (T ./ Δt + (1.0 - θimp) .* Y) * ι.coeffs
        else
            b = (T  * ι.coeffs) ./ Δt
        end

        # Non-inductive source
        Sni !== nothing && (b += Sni)

        # On-axis boundary condition
        b[1] = 0.0

        # Edge boundary condtions
        if Vedge !== nothing
            # Constant loop voltage
            # α d(β*ι)/dρ = Vedge at ρ=1
            b[end] = Vedge
            Sni !== nothing && (b[end] += Vni(QI, η, 1.0))
        else
            # Constant ι (i.e., current)
            if Ip === nothing
                # Match initial current
                b[end] = QI.ι(1.0)
            else
                # Match desired current
                b[end] = Ip * μ₀ * (2π)^2 / (QI.dΦ_dρ(1.0) * QI.dV_dρ(1.0) * QI.fsa_∇ρ²_R²(1.0))
            end
        end
        
        c = invA * b
        ι = FE_rep(ρ, c)
        
        if debug && ((mod(n, Np) == 0) || (n == Nt))
            np += 1
            ιs[np] = deepcopy(ι)
            times[np] = round(Δt*n, digits=3)
        end
        
    end

    JtoR = Jt_R(QI, ι=ι)

    debug && plot_JtoR_profiles(QI, ιs, times, Ncol)

    return QED_state(QI, ι, JtoR)

end

function steady_state(QI::QED_state, η;
                      Y = nothing,
                      Vedge = nothing, Ip = nothing,
                      debug = false)
    Y === nothing && (Y = define_Y(QI, η))
    return diffuse(QI, η, Inf, 1, θimp=1.0, T=0.0.*Y, Y=Y, Vedge=Vedge, Ip=Ip, debug=debug)
end