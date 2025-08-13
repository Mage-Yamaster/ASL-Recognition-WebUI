import pyfiglet
import sys
import os

# RGB color text
def printr(text, r, g, b):
    textp = text+"\n"
    print(f'\033[38;2;{r};{g};{b}m{textp}\033[0m', end="")

# getBigText
def getBig(text):
    return pyfiglet.figlet_format(text)

# get Pipe box
def getPipeBox(text, offset, width, height):
    border = " " * offset + "-" * (len(text) + width * 2 + 2) + "\n"
    middle = " " * offset + "|" + " " * width + text + " " * width + "|\n"
    side_line = " " * offset + "|" + " " * (len(text) + width * 2) + "|\n"
    box = border
    if height > 1:
        box += side_line * width
        box += middle
        box += side_line * width
    else:
        box += middle
    box += border

    return box

def truncate_float(number, decimals):
    """
    Truncates a float to a specified number of decimal places.

    Args:
        number (float): The float to truncate.
        decimals (int): The number of decimal places to keep.

    Returns:
        float: The truncated float.
    """
    factor = 10 ** decimals
    return int(number * factor) / factor