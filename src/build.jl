struct Waveform{T<:Real}
    f::Function
end

function (W::Waveform{T})(t::Real)::T where T
    return W.f(t)
end

# Constant waveform
function Waveform(value::T) where {T<:Real}
    return Waveform{T}(t -> value)
end

# Interpolated waveform
function Waveform(ts::AbstractVector{T}, values::AbstractVector{S}) where {T<:Real, S<:Real}
    TS = promote_type(T, S)
    fitp = DataInterpolations.CubicSpline(values, ts; extrapolate=true)
    return Waveform{TS}(t -> fitp(t))
end

# Functional waveform
# User should directly call Waveform{T}(f) as desired

mutable struct QED_build{T<:Real, MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                         VR1<:AbstractVector{<:Real}, VR2<:AbstractVector{<:Real}, VR3<:AbstractVector{<:Real},
                         VWF<:Vector{<:Waveform}}
    Ic::VR1
    Vc::VR1
    Rc::VR2
    Mcc::MR1
    Vni::T
    Rp::T
    Lp::T
    Mpc::VR2
    dMpc_dt::VR2
    V_waveforms::VWF
    _A::MR2
    _b::VR3
    function QED_build(Ic::VR1, Vc::VR1, Rc::VR2,
                       Mcc::MR1, Vni::T, Rp::T, Lp::T, Mpc::VR2, dMpc_dt::VR2,
                       V_waveforms::VWF, A::MR2, b::VR3) where {T<:Real, MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
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
        return new{T, MR1, MR2, VR1, VR2, VR3, VWF}(Ic, Vc, Rc, Mcc, Vni, Rp, Lp, Mpc, dMpc_dt, V_waveforms, A, b)
    end
end

function QED_build(Ic::VR1, Vc::VR1, Rc::VR2, Mcc::MR1, V_waveforms::VWF) where {MR1<:AbstractMatrix{<:Real},
                                                                                 VR1<:AbstractVector{<:Real},VR2<:AbstractVector{<:Real},
                                                                                 VWF<:Vector{<:Waveform}}
    Mpc = zero(Rc)
    dMpc_dt = zero(Rc)
    return QED_build(Ic, Vc, Rc, Mcc, 0.0, 0.0, 0.0, Mpc, dMpc_dt, V_waveforms)
end

function QED_build(Ic::VR1, Vc::VR1, Rc::VR2, Mcc::MR1,
                   Vni::T, Rp::T, Lp::T, Mpc::VR2, dMpc_dt::VR2, V_waveforms::VWF) where {T<:Real, MR1<:AbstractMatrix{<:Real},
                                                                                          VR1<:AbstractVector{<:Real}, VR2<:AbstractVector{<:Real},
                                                                                          VWF<:Vector{<:Waveform}}
    Nc = length(Ic)
    A = zeros(promote_type(Float64, T, eltype(VR2), eltype(MR1)), Nc, Nc)
    b = zeros(promote_type(Float64, T, eltype(VR1), eltype(VR2), eltype(MR1)), Nc)
    return QED_build(Ic, Vc, Rc, Mcc, Vni, Rp, Lp, Mpc, dMpc_dt, V_waveforms, A, b)
end

function update_voltages!(build::QED_build, t::Real)
    Vc, WF = build.Vc, build.V_waveforms
    for k in eachindex(Vc)
        Vc[k] = WF[k](t)
    end
    return build
end