def milliseconds_to_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60

    remaining_seconds = seconds % 60
    remaining_minutes = minutes % 60
    remaining_hours = hours % 24

    time_string = "{:02d}:{:02d}:{:02d}".format(remaining_hours, remaining_minutes, remaining_seconds)

    return time_string