# QED.jl

Q Evolution by Diffusion

This evolves the rotational transform ι (inverse of the safety factor q) in time according to a loop voltage.

Equations implemented can be found in [equations_readme.pdf](equations_readme.pdf) and borrow extensively from:
Jardin, Stephen. *Computational Methods in Plasma Physics*. CRC Press, 2010.

## IMAS current profiles

`initialize(dd; j_tor_from=:core_profiles)` imports the toroidal current on
the core-profile grid. When `core_profiles.profiles_1d[].j_total` is explicitly
stored, it also imports the equivalent parallel current, with
`<J⋅B> = B₀ * j_total`. At initialization, `Jt_R` and `JB` reproduce these
reference profiles on their native nodes. Repeated initialization and current
writeback at fixed equilibrium therefore preserve an IMAS-consistent input
pair without requiring time evolution.

Both current reconstruction and resistive evolution use the same linear
operator. If `iota_ref` is the imported rotational transform, define
`r = iota / iota_ref` and
`A = B₀ * (dΡ/dρ)^2 * ρ * <|∇ρ|²/R²> / μ₀`. The currents are

```
<Jt/R> = r * <Jt/R>_ref + A * (iota' - r * iota_ref')
<J⋅B>  = r * <J⋅B>_ref  + F * A * (iota' - r * iota_ref')
```

This form obtains the coefficient of `iota` from the supplied current instead
of differentiating separately interpolated geometry. The derivative coefficient
is unchanged. With continuously consistent reference data, it recovers the
original current operator. The diffusion matrix uses this same `<J⋅B>` in Ohm's
law, including steady-state and edge-loop-voltage solves. The reference transform
satisfies the axis regularity condition imposed during evolution, avoiding a
finite projection on the first time step.

Array callers can supply `ρ_j_tor=(rho, jtor_over_r)` and
`ρ_j_parallel=(rho, jparallel)` to use this representation. Without a supplied
parallel current, or with `j_tor_from=:equilibrium`, the existing reconstruction
from geometry derivatives is retained. The caller remains responsible for
consistent toroidal and parallel input profiles. This preserves an input state;
it does not reconcile inconsistent equilibrium data or make finite-time runs
independent of grid resolution and restart frequency. The total-current boundary
condition retains its existing definition through edge poloidal flux.

## Online documentation
For more details, see the [online documentation](https://projecttorreypines.github.io/QED.jl/dev).

![Docs](https://github.com/ProjectTorreyPines/QED.jl/actions/workflows/make_docs.yml/badge.svg)
