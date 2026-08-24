module StageTimings

export print_stage_timings, time_stage!

function time_stage!(f::F, timings::Vector{Pair{String, Float64}}, name::String) where {F}
    """
    Run a computation once, save its elapsed time, and return its result.

    # Arguments:
    - f::F: The computation to time.
    - timings::Vector{Pair{String,Float64}}: List to which the stage timing is added.
    - name::String: Name of the timed stage.

    # Returns:
    - value: The value returned by f.
    """

    # @timed performs one real execution.
    timing = @timed f()
    push!(timings, name => timing.time)
    return timing.value
end

function print_stage_timings(title::String, timings::Vector{Pair{String, Float64}})
    """
    Print the elapsed time and percentage of the measured total for each stage.

    # Arguments:
    - title::String: Heading printed above the stage breakdown.
    - timings::Vector{Pair{String,Float64}}: Stage names and elapsed times in seconds.

    # Returns:
    - Nothing.
    """

    total = sum(last, timings)
    println("\n$(title) ($(round(total, digits = 1)) s total):")
    for (name, seconds) in timings
        percentage = total > 0 ? round(Int, 100 * seconds / total) : 0
        println("  $(rpad(name, 42)) $(lpad(round(seconds, digits = 1), 7)) s  " *
                "$(lpad(percentage, 3))%")
    end
    return nothing
end

end
