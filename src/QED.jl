__precompile__()

module QED

using FiniteElementHermite
using BandedMatrices
import ForwardDiff
import JSON
using ArgParse
import LinearAlgebra: mul!, rmul!, Diagonal

using Requires

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

const μ₀ = 4e-7 * π

include("build.jl")

include("initialize.jl")
export from_imas, η_imas, η_FE, η_mock, initialize

include("current.jl")
export Jt_R, JB, Ip

include("diffuse.jl")
export diffuse, steady_state

include("app.jl")

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__, all=false, imported=false) if name != Symbol(@__MODULE__)]

end
