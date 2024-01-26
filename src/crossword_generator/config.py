from dataclasses import dataclass

import numpy as np


@dataclass
class DefaultArguments:
    # Layout Handler
    PATH_TO_LAYOUT: str = None
    NUM_ROWS: int = 4
    NUM_COLS: int = 5

    # Word Handler
    PATH_TO_WORDS: str = None
    MIN_WORD_LENGTH: int = 3
    MAX_NUM_WORDS: int = np.inf

    # Runtime arguments
    MAX_MCTS_ITERATIONS: int = 1000
    MAX_GENERAL_ITERATIONS: int = 1
    NUM_BLOCKS_TO_ADD_IF_UNSUCCESSFUL: int = 0
    SYMMETRIC_DESIGN: bool = False

    # General arguments
    RANDOM_SEED: int = 123
    OUTPUT_PATH = None


@dataclass
class Constants:
    EMPTY_SYMBOL = "_"
    EMPTY_SYMBOL_FOR_PATTERN_MATCHING = "[A-Z]"
    BLOCK_SYMBOL = ""
    BLOCK_SEPARATOR = "$"

    ROW_INDEX = 0
    COL_INDEX = 1
    WORD_COL_NAME = "answer"
