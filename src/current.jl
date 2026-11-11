
"""
    Jt_R(QI::QED_state; ι::FE_rep=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)

For a given QED_state `QI`, return <Jt/R> on grid `ρ` with rotational transform `ι`
By default, use the grid and rotational transform in `QI` itself
"""
function Jt_R(QI::QED_state; ι::FE_rep=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)
    J = zero(ρ)
    return Jt_R!(J, QI; ι, ρ)
end

"""
    Jt_R!(J::AbstractVector{<:Real}, QI::QED_state; ι::FE_rep=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)

For a given QED_state `QI`, compute <Jt/R> on grid `ρ` with rotational transform `ι` and store in-place in `J`
By default, use the grid and rotational transform in `QI` itself
"""
function Jt_R!(J::AbstractVector{<:Real}, QI::QED_state; ι::FE_rep=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)
    if QI.current_reference !== nothing
        for k in eachindex(ρ)
            ratio, change = current_change(QI, ι, ρ[k])
            J[k] = ratio * QI.current_reference.JtoR(ρ[k]) + change
        end
        return J
    end
    χ(x) = x * ι(x) * fsa_∇ρ²_R²(QI, x)
    dχ(x) = ForwardDiff.derivative(χ, x)

    Threads.@threads for k in eachindex(ρ)
        x = ρ[k]
        if x == 0
            γ = 1.0
        else
            γ = x * D(QI.dV_dρ, x) / QI.dV_dρ(x)
        end
        J[k] = dχ(x) + ι(x) * fsa_∇ρ²_R²(QI, x) * γ
    end
    J .*= QI.B₀ * QI.dΡ_dρ^2 / μ₀
    return J
end

# Factor the current operator around the imported equilibrium transform.
# In the continuous equations Jt/R = A*ι' + Bt*ι and J.B = F*A*ι' + Bp*ι.
# Using the imported reference currents to recover Bt and Bp avoids
# differentiating separately interpolated geometry to reconstruct them.
@inline current_derivative_coefficient(QI::QED_state, x::Real) =
    QI.B₀ * QI.dΡ_dρ^2 * x * fsa_∇ρ²_R²(QI, x) / μ₀

@inline function current_change(QI::QED_state, ι, x::Real)
    ratio = ι(x) / QI._ι_eq(x)
    change = current_derivative_coefficient(QI, x) * (D(ι, x) - ratio * D(QI._ι_eq, x))
    return ratio, change
end

"""
    JB(QI::QED_state; ι=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)

For a given QED_state `QI`, return <J⋅B> on grid `ρ` with rotational transform `ι`
By default, use the grid and rotational transform in `QI` itself
"""
function JB(QI::QED_state; ι=QI.ι, ρ::AbstractVector{<:Real}=QI.ρ)
    if QI.current_reference !== nothing
        current = zero(ρ)
        for k in eachindex(ρ)
            ratio, change = current_change(QI, ι, ρ[k])
            current[k] = ratio * QI.current_reference.JB(ρ[k]) + QI.F(ρ[k]) * change
        end
        return current
    end
    fsa_JB = QI.F.(ρ) .* Jt_R(QI; ι, ρ)
    return fsa_JB .- D.(Ref(QI.F), ρ) .* dΦ_dρ.(Ref(QI), ρ) .* fsa_∇ρ²_R².(Ref(QI), ρ) .* ι.(ρ) ./ (2π * μ₀)
end

"""
    Ip(QI::QED_state)

Return the total plasma current for QED_state `QI`
"""
function Ip(QI::QED_state)
    return dΦ_dρ(QI, 1) * QI.dV_dρ(1) * fsa_∇ρ²_R²(QI, 1) * QI.ι(1) / ((2π)^2 * μ₀)
end

"""
    Vni(QI::QED_state, η, x)

Return the loop voltage associated with the non-inductive current for QED_state `QI`
with (callable) resitivity `η` at `ρ=x`
"""
@inline function Vni(QI::QED_state, η::F, x) where {F}
    return 2π * η(x) * QI.JBni(x) / (QI.F(x) * QI.fsa_R⁻²(x))
end
