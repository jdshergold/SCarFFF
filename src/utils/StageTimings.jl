module StageTimings

using Printf

export print_stage_timings, time_stage!, print_heading, elapsed_time, summarise_timings

const OUTPUT_WIDTH = 68

function print_heading(stage, label, number, count)
    """
    Print the heading for one molecule or crystal in a calculation stage.

    # Arguments:
    - stage::String: The calculation stage.
    - label::String: The molecule or crystal label.
    - number::Int: The system number.
    - count::Int: The number of systems.
    """

    println('\n', "="^OUTPUT_WIDTH)
    println("$(stage)  [$(number)/$(count)]  $(label)")
    println("="^OUTPUT_WIDTH)
    flush(stdout)
end

function elapsed_time(seconds)
    """
    Format an elapsed time in seconds without rounding a short calculation to zero.

    # Arguments:
    - seconds::Real: The elapsed time in seconds.

    # Returns:
    - String: The time with its unit.
    """

    return seconds < 0.05 ? "<0.1 s" : @sprintf("%.1f s", seconds)
end

function summarise_timings(timings, groups)
    """
    Collect measured substages into the named rows of a timing summary.

    # Arguments:
    - timings::Vector: The measured names and times.
    - groups::Vector: Each summary name and its substage names.

    # Returns:
    - Vector: The summary names and times for stages that were performed.
    """

    summary = Pair{String, Float64}[]
    for (name, stages) in groups
        measured = [seconds for (stage, seconds) in timings if stage in stages]
        isempty(measured) || push!(summary, name => sum(measured))
    end
    return summary
end

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

function print_stage_timings(title::String, timings::Vector{Pair{String, Float64}};
                             total = sum(last, timings), column = "Time")
    """
    Print the elapsed time for each stage and the total.

    # Arguments:
    - title::String: Heading printed above the stage breakdown.
    - timings::Vector{Pair{String,Float64}}: Stage names and elapsed times in seconds.
    - total::Real: The total elapsed time in seconds.
    - column::String: The time column heading.

    # Returns:
    - Nothing.
    """

    return print_stage_timings(title, [timings], [column]; totals = [total])
end

function print_stage_timings(title::String, timings::Vector{Vector{Pair{String, Float64}}},
                             columns::Vector{String}; totals = [sum(last, stages) for stages in timings])
    """
    Print stage times side by side for each conformer.

    # Arguments:
    - title::String: The table heading.
    - timings::Vector: Each conformer's stage names and times.
    - columns::Vector{String}: The conformer headings.
    - totals::Vector: The total elapsed time for each conformer, in seconds.
    """

    names = unique([name for stages in timings for (name, _) in stages])
    name_width = max(length(title), maximum(length, names; init = 5), 36)
    column_width = max(maximum(length, columns; init = 4), 10) + 2
    println("\n  ", rpad(title, name_width), join(lpad.(columns, column_width)))
    println("  ", "-"^(name_width + column_width * length(columns)))
    for name in names
        print("  ", rpad(name, name_width))
        for stages in timings
            index = findfirst(stage -> first(stage) == name, stages)
            value = index === nothing ? "—" : elapsed_time(last(stages[index]))
            print(lpad(value, column_width))
        end
        println()
    end
    println("  ", rpad("Total", name_width), join(lpad.(elapsed_time.(totals), column_width)))
    flush(stdout)
    return nothing
end

end
