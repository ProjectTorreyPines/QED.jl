struct Waveform{T<:Real}
    f::Function
end

(W::Waveform{T})(t::Real)::T = W.f(t)

# Constant waveform
function Waveform(value::T) where {T<:Real}
    return Waveform{T}(t -> value)
end

# Interpolated waveform
function Waveform(ts::AbstractVector{T}, values::AbstractVector{S}; order=:cubic) where {T<:Real, S<:Real}
    TS = promote_type(T, S)
    fitp = IMAS.interp1d(ts, values, order)
    return Waveform{TS}(t -> fitp(t))
end

# Functional waveform
# User should directly call Waveform{T}(f) as desired

mutable struct QED_build{MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                         VR1<:AbstractVector{<:Real}, VR2<:AbstractVector{<:Real}, VR3<:AbstractVector{<:Real},
                         VWF<:Vector{<:Waveform}}
    Ic::VR1
    Vc::VR1
    Rc::VR2
    Mcc::MR1
    Mpc::VR2
    dMpc_dt::VR2
    V_waveforms::VWF
    _A::MR2
    _b::VR3
    function QED_build(Ic<:VR1, Vc<:VR1, Rc<:VR2,
                       Mcc<:MR1, Mpc::VR2, dMpc_dt<:VR2,
                       V_waveforms<:VWF, A::MR2, b::VR3) where {MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                                                                VR1<:AbstractVector{<:Real}, VR2<:AbstractVector{<:Real}, VR3<:AbstractVector{<:Real},
                                                                VWF<:Vector{<:Waveform}}
        Nc = length(Ic)
        @assert length(Vc) === Nc
        @assert length(Rc) === Nc
        @assert size(Mcc) === (Nc, Nc)
        @assert length(Mpc) === Nc
        @assert length(dMpc_dt) === Nc
        @assert length(V_waveforms) === Nc
        @assert size(A) === (Nc, Nc)
        @assert length(b) === Nc
        return new{MR1, MR2, VR1, VR2, VR3, VWF}(Ic, Vc, Rc, Mcc, Mpc, dMpc_dt, V_waveforms, A, b)
    end
end

function QED_build(Ic<:VR1, Vc<:VR1, Rc<:VR2, Mcc<:MR1, Mpc::VR2, dMpc_dt<:VR2, V_waveforms<:VWF) where {MR1<:AbstractMatrix{<:Real},
                                                                                                         VR1<:AbstractVector{<:Real}, VR2<:AbstractVector{<:Real},
                                                                                                         VWF<:Vector{<:Waveform}}
    Nc = length(Ic)
    A = zeros(promote_type(Float64, eltype(VR2), eltype(MR1)), Nc, Nc)
    b = zeros(promote_type(Float64, eltype(VR1), eltype(VR2), eltype(MR1)), Nc)
    return QED_build(Ic, Vc, Rc, Mcc, Mpc, dMpc_dt, V_waveforms, A, b)
end

function update_voltages!(build::QED_build, t::Real)
    Vc, WF = build.Vc, build.V_waveforms
    for k in eachindex(Vc)
        Vc[k] = WF[k](t)
    end
    return QB
end

# This does evolution of the build currents, assuming no plasma
function evolve!(build::QED_build, tmax::Real, Nt::Integer; θimp::Real=0.5)
    Ic = build.Ic

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
    end
    return build
end

function build_matrix!(build::QED_build, inv_Δt::Real, θimp::Real)
    Rc, Mcc, A = build.Rc, build.Mcc, build.A
    A .= inv_Δt .* Mcc
    (θimp != 0.0) && (A.+= θimp .* Diagonal(Rc))
    return A
end

function build_rhs!(build::QED_build, t, inv_Δt::Real, θexp::Real)
    Ic, Vc, Rc, Mcc, b = build.Ic, build.Vc, build.Rc, build.Mcc, build.b

    # explicit inductive and resistive terms
    mul!(b, Mcc, Ic)
    b .*= inv_Δt
    (θexp != 0.0) && (b .-= θexp .* Rc .* Ic)

    # source voltage at requested time
    update_voltages!(build, t)
    b += Vc
    return b
end