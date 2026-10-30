"""Shared headings and timing tables for calculation output."""

OUTPUT_WIDTH = 68


def print_heading(stage, label, number, count):
    """
    Print the heading for one molecule or crystal in a calculation stage.

    # Arguments:
    - stage::str: The calculation stage.
    - label::str: The molecule or crystal label.
    - number::int: The system number.
    - count::int: The number of systems.
    """

    print('\n' + '=' * OUTPUT_WIDTH)
    print(f'{stage}  [{number}/{count}]  {label}')
    print('=' * OUTPUT_WIDTH, flush=True)


def print_stage_timings(title, timings, total):
    """
    Print the elapsed time for each stage and the total.

    # Arguments:
    - title::str: The table heading.
    - timings::list: The stage names and elapsed times, in seconds.
    - total::float: The total elapsed time, in seconds.
    """

    name_width = max(len(title), max((len(name) for name, _ in timings), default=5), 36)
    print(f'\n  {title:<{name_width}}{"Time":>12}')
    print('  ' + '-' * (name_width + 12))
    for name, seconds in [*timings, ('Total', total)]:
        elapsed = '<0.1 s' if seconds < .05 else f'{seconds:.1f} s'
        print(f'  {name:<{name_width}}{elapsed:>12}', flush=True)
