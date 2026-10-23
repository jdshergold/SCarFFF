# Build the HDF5 files that test_job_io.py reads back, using the real write_results rather than a
# Python copy of the layout. That is the only way to catch an axis order, dtype or units mismatch
# between the two languages. This is not a test suite of its own, so runtests.jl does not include
# it. test_job_io.py runs it with the output path as its one argument.

using HDF5
include("../src/jobs/Results.jl")
using .MolecularResults

path = ARGS[1]

# The spherical grid every transition shares. The three axes are deliberately different lengths, so
# a permuted axis cannot hide behind a square array.
grid = Dict("q" => (Float32[0, 0.5, 1], "keV"),
            "theta" => (Float32.(range(0, π; length=4)), "rad"),
            "phi" => (Float32.(range(0, 2π; length=5)), "rad"))

# Each entry spells out its own four indices as decimal digits, in the writer's axis order
# (transition, q, θ, ϕ). So f_s[1,3,2,4] is 1324, and if Python hands back the axes in the wrong
# order the digits come out jumbled and say which axes were swapped. The imaginary part is minus the
# real part, so confusing the two parts shows up as a sign.
f_s = ComplexF32[complex(1000a + 100b + 10c + d, -(1000a + 100b + 10c + d))
                 for a in 1:2, b in 1:3, c in 1:4, d in 1:5]

# The same digit trick for the intensity coefficients, indexed (transition, q, spherical harmonic).
coefficients = Float32[100a + 10b + c for a in 1:2, b in 1:3, c in 1:9]

# The three Cartesian V tensors, which sit outside the transitions because all of them share one
# set. Only that they arrive is checked, so the values are all ones.
V = (ones(ComplexF32, 2, 3), ones(ComplexF32, 2, 4), ones(ComplexF32, 2, 5))

# Save transitions 2 and 5, so the original indices have to survive rather than being renumbered.
write_results(path, "spherical", "fixture", grid, [2, 5], Float32[1.23, 4.56];
              f_s=f_s, f_sq_lm=coefficients, V_tensors=V)

# A failed replacement must leave the completed calculation intact.
# One energy for two transitions, so the write runs out of energies half way through.
try
    write_results(path, "spherical", "incomplete", grid, [2, 5], Float32[1.23]; f_s=f_s)
    error("Expected the incomplete calculation to fail.")
catch error
    error isa BoundsError || rethrow()
end
@assert !isfile(path * ".tmp")

# The same again, but this time the write finishes and the file fails its own check.
try
    write_results(path, "spherical", "unchecked", grid, [2, 5], Float32[1.23, 4.56];
                  f_s=f_s, check_result=temporary -> error("Verification failed."))
    error("Expected verification to fail.")
catch error
    @assert error isa ErrorException && error.msg == "Verification failed."
end
@assert !isfile(path * ".tmp")

# Neither failure should have touched the file written first.
h5open(path, "r") do io
    @assert read(attributes(io)["stage_signature"]) == "fixture"
    @assert read(attributes(io)["completed"])
end

# The Cartesian fixture, on a box grid whose three axes again have different lengths. The same f_s
# serves here, since its axes are now read as (transition, q_x, q_y, q_z).
grid_xyz = Dict("q_x" => (Float32[-1, 0, 1], "keV"),
                "q_y" => (Float32[-2, -1, 1, 2], "keV"),
                "q_z" => (Float32[-3, -1, 0, 1, 3], "keV"))
write_results(path * ".cartesian", "cartesian", "fixture", grid_xyz, [2, 5], Float32[1.23, 4.56];
              f_s=f_s, V_tensors=V)

# The Fourier fixture adds the real-space axes that the transition density lives on, so its units
# and precision get checked too.
for (name, count) in zip(("x", "y", "z"), (3, 4, 5))
    grid_xyz[name] = (Float32.(range(-1, 1; length=count)), "angstrom")
end
write_results(path * ".fourier", "fourier", "fixture", grid_xyz, [2, 5], Float32[1.23, 4.56];
              f_s=f_s, transition_density=real.(f_s))
