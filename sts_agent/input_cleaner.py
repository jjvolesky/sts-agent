import io

# thank you to s3dev (https://stackoverflow.com/a/78114529/27856187)
def strip_ansi_colour(text: str) -> iter:
    """Strip ANSI colour sequences from a string.

    Args:
        text (str): Text string to be stripped.

    Returns:
        iter[str]: A generator for each returned character. Note,
        this will include newline characters.

    """
    buff = io.StringIO(text)
    while (b := buff.read(1)):
        if b == '\x1b':
            while (b := buff.read(1)) != '{': continue
        else:
            yield b