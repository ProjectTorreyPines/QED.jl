function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--tmax", "-T"
        help = "Maximum time to diffuse (seconds)"
        arg_type = typeof(0.0)
        default = Inf
        "--timesteps", "-N"
        help = "Number of time steps"
        arg_type = typeof(0)
        default = 1
        "--Vedge", "-V"
        help = "Edge loop voltage boundary condition (Volts)"
        arg_type = typeof(0.0)
        "--Ip", "-I"
        help = "Total plasma current boundary condition (Amps)"
        arg_type = typeof(0.0)
        """
        --timeslice
        """
        help = "Time slice to use in JSON file"
        arg_type = typeof(0)
        default = 1
        """
        input_file
        """
        help = "Input JSON filename"
        required = true
        """
        output_file
        """
        help = "Output JSON filename"
        default = "qed_output.json"
        required = false
    end

    return parse_args(s)
end

function julia_main()::Cint
    println("QED - Q Evolution by Diffusion")

    args = parse_commandline()
    input = JSON.parsefile(args["input_file"])
    timeslice = args["timeslice"]

    QI = from_imas(input, timeslice)
    η = η_imas(input, timeslice)

    function printbc()
        if args["Ip"] !== nothing
            return "Ip = $(args["Ip"]) A"
        elseif args["Vedge"] !== nothing
            return "Vedge = $(args["Vedge"]) V"
        else
            return "constant plasma current"
        end
    end

    if args["tmax"] === Inf
        println("  Running in steady-state with " * printbc())
        QO = steady_state(QI, η; Vedge=args["Vedge"], Ip=args["Ip"])
    else
        println("  Running for $(args["tmax"]) s in $(args["timesteps"]) time steps with " * printbc())
        QO = diffuse(QI, η, args["tmax"], args["timesteps"];
            Vedge=args["Vedge"], Ip=args["Ip"])
    end

    println("  Diffusion complete")

    # Write ι and <Jt/R> to the output file
    output = deepcopy(input)
    eqt = output["equilibrium"]["time_slice"][timeslice]
    ρ = eqt["profiles_1d"]["rho_tor"] / eqt["profiles_1d"]["rho_tor"][end]
    eqt["profiles_1d"]["q"] = 1.0 ./ QO.ι.(ρ)
    eqt["profiles_1d"]["j_tor"] = QO.JtoR.(ρ) ./ eqt["profiles_1d"]["gm9"]

    println("  Outputting results to $(args["output_file"])")

    open(args["output_file"], "w") do f
        return JSON.print(f, output, 1)
    end

    println("Exiting QED")

    return 0
end