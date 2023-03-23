def milliseconds_to_time(milliseconds):
    # Calculate the number of seconds, minutes, and hours
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60

    # Calculate the remaining seconds, minutes, and hours
    remaining_seconds = seconds % 60
    remaining_minutes = minutes % 60
    remaining_hours = hours % 24

    # Format the time as a string with milliseconds
    time_string = "{:02d}:{:02d}:{:02d}.{:03d}".format(remaining_hours, remaining_minutes, remaining_seconds, milliseconds % 1000)

    # Return the formatted time string
    return time_string