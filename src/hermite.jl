#==============================================================
Define Hermite cubic finite elements
===============================================================#

# For three points (x[1], y[1]), (x[2], y[2]), and (x[3], y[3]),
#   fit a parabola and find derivative at x[2]
function quad_deriv(x::Tuple{T,T,T}, y::Tuple{T,T,T}) where T <: Real
    hl = x[2] - x[1]
    hu = x[3] - x[2]
    return ((y[3] - y[2])*hl^2 + (y[2] - y[1])*hu^2)/(hu*hl*(hu+hl))
end

# Returns dy/dx at every point in x, based on local quadratic fit
function fit_derivative(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real
    N = length(x)
    dy_dx = zeros(N)
    dy_dx[1] = quad_deriv((x[3], x[1], x[2]), (y[3], y[1], y[2]))
    dy_dx[end] = quad_deriv((x[end-1], x[end], x[end-2]), (y[end-1], y[end], y[end-2]))
    for k in 2:N-1
        dy_dx[k] = quad_deriv(Tuple(x[k-1:k+1]), Tuple(y[k-1:k+1]))
    end
    return dy_dx
end


function νe(x::Real, k::Integer, ρ::AbstractVector{T}) where T <: Real
    if k > 1  && x >= ρ[k-1] && x <= ρ[k]
        # we're in the lower half
        t = (x - ρ[k])/(ρ[k] - ρ[k-1])
        return -2.0*t^3 - 3.0*t^2 + 1.0
    elseif k < length(ρ)  && x >= ρ[k] && x <= ρ[k+1]
        # we're in the upper half
        t = (x - ρ[k])/(ρ[k+1] - ρ[k])
        return 2.0*t^3 - 3.0*t^2 + 1.0
    end
    return 0.0
end
function D_νe(x::Real, k::Integer, ρ::AbstractVector{T}) where T <: Real
    if k > 1  && x >= ρ[k-1] && x <= ρ[k]
        # we're in the lower half
        ihl = 1.0/(ρ[k] - ρ[k-1])
        t = (x - ρ[k]) * ihl
        return -6.0 * ihl * t * (t + 1.0)
    elseif k < length(ρ)  && x >= ρ[k] && x <= ρ[k+1]
        # we're in the upper half
        ihu = 1.0/(ρ[k+1] - ρ[k])
        t = (x - ρ[k]) * ihu
        return 6.0 * ihu * t * (t - 1.0)
    end
    return 0.0
end
function I_νe(x::Real, k::Integer, ρ::AbstractVector{T}) where T <: Real
    k==1 ? hl = 0.0 : hl = ρ[k] - ρ[k-1]
    k==length(ρ) ? hu = 0.0 : hu = ρ[k+1] - ρ[k]
    
    if x <= ρ[k]
        Iu = 0.0
        if k > 1  && x >= ρ[k-1]
            # we're in the lower half
            t = (x - ρ[k])/hl
            Il = hl*(-0.5*t^4 - t^3 + t + 0.5)
        else
            Il = 0.0
        end
    elseif x >= ρ[k]
        Il = 0.5*hl
        if k < length(ρ) && x <= ρ[k+1]
            # we're in the upper half
            t = (x - ρ[k])/hu
            Iu = hu*t*(0.5*t^3 - t^2 + 1.0)
        else
            Iu = 0.5*hu
        end
    end
    return Il + Iu 
end

function νo(x::Real, k::Integer, ρ::AbstractVector{T}) where T <: Real
    if k > 1  && x >= ρ[k-1] && x <= ρ[k]
        # we're in the lower half
        hl = ρ[k] - ρ[k-1]
        t = (x - ρ[k])/hl
        return hl * t * (t + 1.0)^2
    elseif k < length(ρ)  && x >= ρ[k] && x <= ρ[k+1]
        # we're in the upper half
        hu = ρ[k+1] - ρ[k]
        t = (x - ρ[k])/hu
        return hu * t * (t - 1.0)^2
    end
    return 0.0
end
function D_νo(x::Real, k::Integer, ρ::AbstractVector{T}) where T <: Real
    if k > 1  && x >= ρ[k-1] && x <= ρ[k]
        # we're in the lower half
        t = (x - ρ[k])/(ρ[k] - ρ[k-1])
        return 3.0*t^2 + 4.0*t + 1.0
    elseif k < length(ρ)  && x >= ρ[k] && x <= ρ[k+1]
        # we're in the upper half
        t = (x - ρ[k])/(ρ[k+1] - ρ[k])
        return 3.0*t^2 - 4.0*t + 1.0
    end
    return 0.0
end
function I_νo(x::Real, k::Integer, ρ::AbstractVector{T}) where T <: Real
    k==1 ? hl = 0.0 : hl = ρ[k] - ρ[k-1]
    if x <= ρ[k]
        Iu = 0.0
        if k > 1  && x >= ρ[k-1]
            # we're in the lower half
            t = (x - ρ[k])/hl
            Il = hl^2*(0.25*t^4 + (2.0/3.0)*t^3  + 0.5*t^2 - 1.0/12.0)
        else
            Il = 0.0
        end
    elseif x >= ρ[k]
        Il = -hl^2 / 12.0
        k==length(ρ) ? hu = 0.0 : hu = ρ[k+1] - ρ[k]
        if k < length(ρ) && x <= ρ[k+1]
            # we're in the upper half
            t = (x - ρ[k])/hu
            Iu = (hu*t)^2 * (0.25*t^2 - (2.0/3.0)*t + 0.5)
        else
            Iu = hu^2 / 12.0
        end
    end
    return Il + Iu 
end

function hermite_coeffs(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real
    dy_dx = fit_derivative(x, y)
    C = zeros(2*length(x))
    C[1:2:end] .= dy_dx
    C[2:2:end] .= y
    return C
end


struct FE_rep{T<:Real}
    x::AbstractVector{T}
    coeffs::AbstractVector{T}
end
FE(x, y) = FE_rep(x, hermite_coeffs(x,y))

function (Y::FE_rep)(x::Real)
    y = 0.0
    for k in 1:length(Y.x)
        y += Y.coeffs[2k-1]*νo(x, k, Y.x)
        y += Y.coeffs[2k]  *νe(x, k, Y.x)
    end
    return y
end
function D(Y::FE_rep, x::Real)
    dy_dx = 0.0
    for k in 1:length(Y.x)
        dy_dx += Y.coeffs[2k-1]*D_νo(x, k, Y.x)
        dy_dx += Y.coeffs[2k]  *D_νe(x, k, Y.x)
    end
    return dy_dx
end
function I(Y::FE_rep, x::Real)
    yint = 0.0
    for k in 1:length(Y.x)
        yint += Y.coeffs[2k-1]*I_νo(x, k, Y.x)
        yint += Y.coeffs[2k]  *I_νe(x, k, Y.x)
    end
    return yint
end

function add_point!(Y::FE_rep, x::Real, y::Real, dydx::Real)
    i = searchsortedfirst(Y.x, x)

    @assert x != Y.x[i] "$(x) already contained in Y.x at index $(i)"

    insert!(Y.x, i, x)
    insert!(Y.coeffs, 2i-1, y)
    insert!(Y.coeffs, 2i-1, dydx)
    return Y
end

function delete_point!(Y::FE_rep, i::Integer)
    deleteat!(Y.x, i)
    deleteat!(Y.coeffs, 2i-1)
    deleteat!(Y.coeffs, 2i-1)
    return Y
end

function resample(Y::FE_rep, x::AbstractVector)
    C = zeros(2*length(x))
    C[2:2:end] .= Y.(x)
    C[1:2:end] .= D.(Ref(Y), x)
    return FE_rep(x, C)
end

#==============================================================
Define inner products
===============================================================#

function nu_nu(x, nu1, k1, nu2, k2, ρ)
    return nu1(x, k1, ρ) * nu2(x, k2, ρ)
end

function f_nu_nu(x, f, nu1, k1, nu2, k2, ρ)
    return f(x) * nu1(x, k1, ρ) * nu2(x, k2, ρ)
end

function nu_fnu_gnu(x, nu1, k1, f, fnu2, g, gnu2, k2, ρ)
    return nu1(x, k1, ρ) * (f(x) * fnu2(x, k2, ρ) + g(x) * gnu2(x, k2, ρ))
end


function f_nu(x, f, nu, k, ρ)
    return f(x) * nu(x, k, ρ)
end

const gξ, gw = gauss(5, -1.0, 1.0)

function integrate(f, lims, tol)
    return quadgk(f, lims..., atol=tol, rtol=sqrt(tol))[1]
    # Nint = length(lims) - 1
    # I = 0.0
    # for i in 1:Nint
    #     #x, w = gauss(7, lims[i], lims[i+1])
    #     dxdξ = 0.5*(lims[i+1] - lims[i])
    #     I += sum(f.(dxdξ .* gξ .+ 0.5*(lims[i]+lims[i+1])) .* gw) .* dxdξ
    # end
    # return I
end

function inner_product(nu1, k1, nu2, k2, ρ)
    if abs(k1 - k2) <= 1
        if k1 != k2
            lims = (min(ρ[k1], ρ[k2]), max(ρ[k1], ρ[k2]))
        elseif k1==1 
            lims = (ρ[1], ρ[2])
        elseif k1==length(ρ)
            lims = (ρ[end-1], ρ[end])
        else
            lims = (ρ[k1-1], ρ[k1], ρ[k1+1])
        end
        tol = eps(typeof(ρ[1]))
        return integrate(x -> nu_nu(x, nu1, k1, nu2, k2, ρ), lims, tol)
    end
    return 0.0
end

function inner_product(f, nu1, k1, nu2, k2, ρ)
    if abs(k1 - k2) <= 1
        if k1 != k2
            lims = (min(ρ[k1], ρ[k2]), max(ρ[k1], ρ[k2]))
        elseif k1==1 
            lims = (ρ[1], ρ[2])
        elseif k1==length(ρ)
            lims = (ρ[end-1], ρ[end])
        else
            lims = (ρ[k1-1], ρ[k1], ρ[k1+1])
        end
        tol = eps(typeof(ρ[1]))
        return integrate(x -> f_nu_nu(x, f, nu1, k1, nu2, k2, ρ), lims, tol)
    end
    return 0.0
end

function inner_product(nu1, k1, f, fnu2, g, gnu2, k2, ρ)
    if abs(k1 - k2) <= 1
        if k1 != k2
            lims = (min(ρ[k1], ρ[k2]), max(ρ[k1], ρ[k2]))
        elseif k1==1 
            lims = (ρ[1], ρ[2])
        elseif k1==length(ρ)
            lims = (ρ[end-1], ρ[end])
        else
            lims = (ρ[k1-1], ρ[k1], ρ[k1+1])
        end
        tol = eps(typeof(ρ[1]))
        return integrate(x -> nu_fnu_gnu(x, nu1, k1, f, fnu2, g, gnu2, k2, ρ), lims, tol)
    end
    return 0.0
end

function inner_product(f, nu, k, ρ)
    if k==1 
        lims = (ρ[1], ρ[2])
    elseif k==length(ρ)
        lims = (ρ[end-1], ρ[end])
    else
        lims = (ρ[k-1], ρ[k], ρ[k+1])
    end
    tol = eps(typeof(ρ[1]))
    return integrate(x -> f_nu(x, f, nu, k, ρ), lims, tol)
end