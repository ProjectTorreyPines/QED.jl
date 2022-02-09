__precompile__()

module QED

using Plots
using QuadGK
using BandedMatrices
using ForwardDiff
using JSON

const μ₀ = 4e-7*π

include("hermite.jl")
export FE_rep, FE, D, I

include("initialize.jl")
export QED_state, parse_imas, from_imas, η_imas, η_mock

include("current.jl")
export Jt_R, JB, Ip

include("diffuse.jl")
export diffuse, steady_state, define_T, define_Y

end
