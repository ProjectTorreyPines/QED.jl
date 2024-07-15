import Pkg
Pkg.activate(".")

import PackageCompiler

const build_dir = @__DIR__
const package_dir = dirname(build_dir)
if length(ARGS) > 0
    const target_dir = ARGS[1]
else
    const target_dir = build_dir * "/QED"
end

PackageCompiler.create_app(package_dir, target_dir; force=true,
    precompile_execution_file=build_dir * "/generate_precompile.jl")
