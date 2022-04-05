__precompile__()

module QED

using FiniteElementHermite
using BandedMatrices
import ForwardDiff
import JSON
using ArgParse

using Requires

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

const μ₀ = 4e-7*π

include("initialize.jl")
export QED_state, from_imas, η_imas, η_mock, initialize

include("current.jl")
export Jt_R, JB, Ip

include("diffuse.jl")
export diffuse, steady_state, define_T, define_Y

include("app.jl")

end
