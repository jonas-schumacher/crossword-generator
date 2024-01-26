import time
from enum import Enum
from typing import List

from crossword_generator.config import Constants


class Orientation(Enum):
    ACROSS = "across"
    DOWN = "down"


def convert_pattern_for_matching(pattern: List[str]) -> str:
    return "".join(pattern).replace(
        Constants.EMPTY_SYMBOL, Constants.EMPTY_SYMBOL_FOR_PATTERN_MATCHING
    )


def timing_decorator():
    def wrap(func):
        def wrapped_function(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Runtime ({func.__name__}): {duration:.1f}s = {duration/60:.1f}min.")
            return result

        return wrapped_function

    return wrap
