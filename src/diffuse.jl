# Matrix for the time-derivative term (without Δt)
# T_mk = (dΦ_dρ * ν_m * ν_k)
function define_T(QI::QED_state; order::Union{Nothing,Integer}=5)
    DΦ = x -> dΦ_dρ(QI, x)
    # Hermite cubics give septadiagonal matrix
    T = BandedMatrix(Zeros(2length(QI.ρ), 2length(QI.ρ)), (3, 3))
    define_T!(T, DΦ, QI.ρ, order)
    return T
end

function define_T!(T::BandedMatrix, DΦ, ρ::AbstractVector{<:Real}, order::Union{Nothing,Integer})
    Threads.@threads for m in eachindex(ρ)
        Me = 2m
        Mo = Me - 1

        if m > 1
            # T[Mo, Mo-3] = 0.0
            T[Me, Me-3] = inner_product(DΦ, νe, m, νo, m-1, ρ, order)

            T[Mo, Mo-2] = inner_product(DΦ, νo, m, νo, m-1, ρ, order)
            T[Me, Me-2] = inner_product(DΦ, νe, m, νe, m-1, ρ, order)

            T[Mo, Mo-1] = inner_product(DΦ, νo, m, νe, m-1, ρ, order)
        end
        T[Me, Me-1] = inner_product(DΦ, νe, m, νo, m, ρ, order)

        T[Mo, Mo] = inner_product(DΦ, νo, m, νo, m, ρ, order)
        T[Me, Me] = inner_product(DΦ, νe, m, νe, m, ρ, order)

        T[Mo, Mo+1] = inner_product(DΦ, νo, m, νe, m, ρ, order)
        if m < length(ρ)
            T[Me, Me+1] = inner_product(DΦ, νe, m, νo, m+1, ρ, order)

            T[Mo, Mo+2] = inner_product(DΦ, νo, m, νo, m+1, ρ, order)
            T[Me, Me+2] = inner_product(DΦ, νe, m, νe, m+1, ρ, order)

            T[Mo, Mo+3] = inner_product(DΦ, νo, m, νe, m+1, ρ, order)
            # T[Me, Me+3] = 0.0
        end
    end
end

function αβ(x::Real, QI::QED_state, η)
    return η(x) * dΦ_dρ(QI, x) * fsa_∇ρ²_R²(QI, x) / (μ₀ * QI.fsa_R⁻²(x))
end

function αdβ_dρ(x::Real, QI::QED_state, η)

    if x != 0
        γ = x * (D(QI.dV_dρ, x) / QI.dV_dρ(x) - D(QI.F, x) / QI.F(x))
    else
        γ = 1.0
    end
    abp = (1.0 + γ) * fsa_∇ρ²_R²(QI, x) + x * D_fsa_∇ρ²_R²(QI, x)
    return 2π * QI.B₀ * QI.dΡ_dρ^2 * η(x) * abp / (μ₀ * QI.fsa_R⁻²(x))
end

# Matrix for the diffusion term
# Y_mk = (ν_m * d/dρ[α * d/dρ(β * ν_k)])
# This gets integrated by parts to avoid second derivatives of finite elements
function define_Y(QI::QED_state, η; order::Union{Nothing, Integer}=5)

    # transform to single argument functions for inner_product
    ab(x) = αβ(x, QI, η)
    adb_dρ(x) = αdβ_dρ(x, QI, η)

    # Hermite cubics give septadiagonal matrix
    Y = BandedMatrix(Zeros(2length(QI.ρ), 2length(QI.ρ)), (3, 3))

    define_Y!(Y, ab, adb_dρ, QI.ρ, order)

    # Y matrix has an integration by parts, so we need the boundary terms

    # Term at ρ=0 uses fact β(0)=0 and ρ * (F/V') * d(V'/F)/dρ = 1.0
    Y[2, 2] -= adb_dρ(0.0)

    # Term at ρ=1 goes in the 2N row (only element non-zero at ρ=1)
    N = length(QI.ρ)
    Y[2N, 2N] += adb_dρ(1.0) # νe(1.0, N, QI.ρ) is 1.0
    Y[2N, 2N-1] += ab(1.0) * D_νo(1.0, N, QI.ρ)

    return Y
end

function define_Y!(Y::BandedMatrix, ab, adb_dρ,  ρ::AbstractVector{<:Real}, order::Union{Nothing,Integer})
    Threads.@threads for m in eachindex(ρ)
        Me = 2m
        Mo = Me - 1

        if m > 1
            #Y[Mo, Mo-3] = 0.0
            Y[Me, Me-3]  = -inner_product(D_νe, m, ab, D_νo, adb_dρ, νo, m-1, ρ, order)
            Y[Mo, Mo-2]  = -inner_product(D_νo, m, ab, D_νo, adb_dρ, νo, m-1, ρ, order)
            Y[Me, Me-2]  = -inner_product(D_νe, m, ab, D_νe, adb_dρ, νe, m-1, ρ, order)

            Y[Mo, Mo-1]  = -inner_product(D_νo, m, ab, D_νe, adb_dρ, νe, m-1, ρ, order)
        end
        Y[Me, Me-1]  = -inner_product(D_νe, m, ab, D_νo, adb_dρ, νo, m, ρ, order)

        Y[Mo, Mo]  = -inner_product(D_νo, m, ab, D_νo, adb_dρ, νo, m, ρ, order)
        Y[Me, Me]  = -inner_product(D_νe, m, ab, D_νe, adb_dρ, νe, m, ρ, order)

        Y[Mo, Mo+1]  = -inner_product(D_νo, m, ab, D_νe, adb_dρ, νe, m, ρ, order)
        if m < length(ρ)
            Y[Me, Me+1]  = -inner_product(D_νe, m, ab, D_νo, adb_dρ, νo, m+1, ρ, order)

            Y[Mo, Mo+2]  = -inner_product(D_νo, m, ab, D_νo, adb_dρ, νo, m+1, ρ, order)
            Y[Me, Me+2]  = -inner_product(D_νe, m, ab, D_νe, adb_dρ, νe, m+1, ρ, order)

            Y[Mo, Mo+3]  = -inner_product(D_νo, m, ab, D_νe, adb_dρ, νe, m+1, ρ, order)
            #Y[Me, Me+3] = 0.0
        end
    end

    return Y
end

function define_Sni(QI::QED_state, η; order::Union{Nothing,Integer}=5)

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
    inv_Δt = Nt / tmax

    θexp = 1.0 - θimp

    ρ = QI.ρ

    # A = T / Δt - θimp * Y
    A = deepcopy(T)
    rmul!(A, inv_Δt)
    (θimp != 0) && (A .-= θimp .* Y)

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
    Sni = define_Sni(QI, η)

    # invert A matrix only once, outside of time stepping loop
    # We could factor and do linear solve each time,
    #   but this seems faster (because small matrix)
    invA = inv(A)

    c = deepcopy(QI.ι.coeffs)
    b = zero(c)
    btmp = zero(c)

    # We advance c, the coefficients of ι,
    #   but we never actually need ι until the end (except for debugging)
    for n in 1:Nt
        mul!(b, T, c)
        #b .= T * c
        rmul!(b, inv_Δt)
        if θexp != 0.0
            mul!(btmp, Y, c)
            rmul!(btmp, θexp)
            b .+= btmp
        end

        # Non-inductive source
        Sni !== nothing && (b .+= Sni)

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
                b[end] = QI._ι_eq(1.0)
            else
                # Match desired current
                b[end] = Ip * μ₀ * (2π)^2 / (dΦ_dρ(QI, 1.0) * QI.dV_dρ(1.0) * fsa_∇ρ²_R²(QI, 1.0))
            end
        end

        mul!(c, invA, b)

        if debug && ((mod(n, Np) == 0) || (n == Nt))
            np += 1
            ιs[np] = FE_rep(ρ, deepcopy(c))
            times[np] = round(Δt*n, digits=3)
        end

    end

    ι = FE_rep(ρ, c)
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