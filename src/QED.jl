__precompile__()

module QED

using QuadGK
using BandedMatrices
import ForwardDiff
import JSON
using ArgParse
import Optim

using Requires

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

const μ₀ = 4e-7*π

include("hermite.jl")
export fit_derivative, FE_rep, FE, D, D2, I, resample, add_point!, delete_point!

include("initialize.jl")
export QED_state, from_imas, η_imas, η_mock, resample, optknt4
include("current.jl")
export Jt_R, JB, Ip

include("diffuse.jl")
export diffuse, steady_state, define_T, define_Y

include("app.jl")

end
