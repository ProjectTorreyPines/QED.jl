using QED
import QED: JSON, ForwardDiff
using Test

# Current diffusion with dual numbers: derivatives against central finite differences of the Float64 solve

@testset "current diffusion with dual numbers" begin
    data = JSON.parsefile(joinpath(dirname(@__DIR__), "sample", "ods_163303Z26-3000.json"))
    e1 = data["equilibrium"]["time_slice"][1]["profiles_1d"]
    cp = data["core_profiles"]["profiles_1d"][1]
    B0 = Float64(data["equilibrium"]["vacuum_toroidal_field"]["b0"][1])
    col(key) = Float64.(e1[key])

    # s[1] scales the resistivity, s[2] the safety factor of the initial state
    function run(s)
        T = typeof(s[1] * s[2])
        QI = initialize(T.(col("rho_tor")), T(B0), T.(col("gm1")), T.(col("f")), T.(col("dvolume_drho_tor")), col("q") .* s[2], T.(col("j_tor")), T.(col("gm9")))
        η = η_FE(T.(Float64.(cp["grid"]["rho_tor_norm"])), s[1] ./ Float64.(cp["conductivity_parallel"]))
        Q = diffuse(QI, η, 1.0, 100; θimp=0.5, Vedge=0.0)
        return vcat([Q.ι(x) for x in 0.1:0.2:0.9], Ip(Q))
    end

    # the derivatives of the inputs are carried by an outer ForwardDiff sweep, around the derivatives that
    # the solver takes internally
    y0 = run([1.0, 1.0])
    J = ForwardDiff.jacobian(run, [1.0, 1.0])
    @test all(isfinite, J)
    h = 1e-4
    for k in 1:2
        e = [Float64(k == 1), Float64(k == 2)]
        fd = (run(1.0 .+ h .* e) .- run(1.0 .- h .* e)) ./ (2h)
        @test J[:, k] ≈ fd rtol = 1e-4 atol = 1e-6 * maximum(abs, fd)
    end
    @test any(!iszero, J[:, 1]) && any(!iszero, J[:, 2])
end
