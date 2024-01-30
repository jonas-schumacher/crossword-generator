from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from crossword_generator.common import timing_decorator
from crossword_generator.config import DefaultArguments
from crossword_generator.layout_handler import (
    ExistingLayoutHandler,
    NewLayoutHandler,
    LayoutHandler,
)
from crossword_generator.state import get_initial_crossword_state
from crossword_generator.tree_search import MCTS, TreeNode
from crossword_generator.word_handler import (
    DictionaryWordHandler,
    FileWordHandler,
    WordHandler,
)


def generate_crossword(
    path_to_layout: str | None = DefaultArguments.PATH_TO_LAYOUT,
    num_rows: int = DefaultArguments.NUM_ROWS,
    num_cols: int = DefaultArguments.NUM_COLS,
    path_to_words: str | None = DefaultArguments.PATH_TO_WORDS,
    max_num_words: int = DefaultArguments.MAX_NUM_WORDS,
    max_mcts_iterations: int = DefaultArguments.MAX_MCTS_ITERATIONS,
    random_seed: int = DefaultArguments.RANDOM_SEED,
    output_path: str | None = DefaultArguments.OUTPUT_PATH,
) -> None:
    """
    Generate a new crossword by combining words into a rectangular grid.

    Examples
    ----------
    1) generate_crossword() --> run with default arguments
    2) generate_crossword(path_to_words="sample_files/words_*.csv") --> read words from CSV files

    Parameters
    ----------
    path_to_layout: str | None
        - str: read grid layout from CSV file (num_rows / num_cols will be ignored)
        - None: generate new grid with "num_rows" and "num_cols"
    num_rows: int | None:
        - int: number of rows the grid should have
        - None: only possible if path_to_layout is not None
    num_cols: int | None:
        - int: number of columns the grid should have
        - None: only possible if path_to_layout is not None
    path_to_words: str | None
        - str: read words from all CSV files that follow the pattern specified in "path_to_words"
        - None: get words from English dictionary
    max_num_words: int
        Limits the number of words in the dictionary to improve runtime
    max_mcts_iterations: int
        Limits the number of MCTS iterations to decrease computation / memory usage
    random_seed: int
        Seed to initialize the random number generator. Change this to get different solutions.
    output_path: str | None
        If provided, save the final grid and a summary to disk

    Returns
    -------
    None

    """
    print("Run crossword creator with the following arguments:")
    print(f"{100 * '-'}")
    print(f"{path_to_layout = }")
    print(f"{num_rows = }")
    print(f"{num_cols = }")
    print(f"{path_to_words = }")
    print(f"{max_num_words = }")
    print(f"{max_mcts_iterations = }")
    print(f"{random_seed = }")
    print(f"{output_path = }")
    print(f"{100 * '-'}")

    np.random.seed(random_seed)

    # Option #1: Read an existing layout from file
    if path_to_layout is not None:
        print(f"Read an existing layout from file: {path_to_layout}.")
        layout_handler = ExistingLayoutHandler(
            path_to_existing_layout=path_to_layout,
        )
    # Option #2: Create a new layout from scratch
    elif num_rows is not None and num_cols is not None:
        print(
            f"No layout was provided. Create a new layout with shape: ({num_rows},{num_cols})."
        )
        layout_handler = NewLayoutHandler(
            num_rows=num_rows,
            num_cols=num_cols,
        )
    else:
        raise ValueError(
            "If no path to a given layout is provided, num_rows and num_cols must be specified."
        )

    print(layout_handler)
    print(f"{100 * '-'}")

    if path_to_words is not None:
        print(f"Read input words from: {path_to_words}.")
        word_handler = FileWordHandler(
            path_to_words=path_to_words,
            word_lengths=layout_handler.word_lengths,
            max_num_words=max_num_words,
        )
    else:
        print(f"Get input words from NLTK corpus.")
        word_handler = DictionaryWordHandler(
            word_lengths=layout_handler.word_lengths,
            max_num_words=max_num_words,
        )

    print(word_handler)
    print(f"{100 * '-'}")

    solved = False
    iteration = 1

    while not solved and iteration <= DefaultArguments.MAX_GENERAL_ITERATIONS:
        print(f"{30 * '-'} Iteration {iteration} {30 * '-'}")
        solved, final_grid, final_statistics = fill_current_layout(
            layout_handler=layout_handler,
            word_handler=word_handler,
            iteration_limit=max_mcts_iterations,
        )
        iteration += 1

        # Optionally save current solution to disk
        if output_path is not None:
            print(f"Persist results to: {output_path}")
            now = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
            Path(output_path).mkdir(parents=True, exist_ok=True)
            for lang, sep in zip(["en", "de"], [",", ";"]):
                final_statistics.to_csv(f"{output_path}/{now}_statistics_{lang}.csv", sep=sep)
                final_grid.to_csv(f"{output_path}/{now}_layout_{lang}.csv", sep=sep)

        # Optionally add some more blocks to the layout after each unsuccessful attempt:
        if DefaultArguments.NUM_BLOCKS_TO_ADD_IF_UNSUCCESSFUL > 0:
            layout_handler = layout_handler.add_blocks_to_layout(
                num_blocks=DefaultArguments.NUM_BLOCKS_TO_ADD_IF_UNSUCCESSFUL
            )
            print(layout_handler)

    if solved:
        print(
            f"Found solution with {layout_handler.num_blocks}/{layout_handler.num_cells} "
            f"= {layout_handler.share_of_blocks:.1%} blocks."
        )
    else:
        print("Failed to find a solution.")


@timing_decorator()
def fill_current_layout(
    layout_handler: LayoutHandler,
    word_handler: WordHandler,
    iteration_limit: int,
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Fill a given layout provided by layout_handler with words provided by word_handler

    Parameters
    ----------
    layout_handler: LayoutHandler
        Contains all relevant information about
        - layout that is supposed to get filled with words
        - relationship between the entries

    word_handler: WordHandler
        Contains all relevant information about
        - words that can be used to fill the entries

    iteration_limit: int
        Specifies how many MCTS searches are started to fill each entry

    Returns
    -------
    Tuple
        bool: True if whole grid has been filled successfully
        pd.DataFrame: final grid
        pd.DataFrame: statistics about final grid
    """

    # Set up MCTS Tree
    searcher = MCTS(
        layout_handler=layout_handler,
        iteration_limit=iteration_limit,
        exploration_constant=layout_handler.num_entries,
    )
    # Set up root state
    current_state = get_initial_crossword_state(
        layout_handler=layout_handler,
        word_handler=word_handler,
    )

    # Optionally fix the first X words of the crossword:
    words_to_fill = []

    for word in words_to_fill:
        current_state = current_state.take_action(
            action=word,
        )

    current_generation = len(current_state.filled_entries)

    # Set up root node
    current_node = TreeNode(
        parent=None,
        action_leading_here=None,
        state=current_state,
    )

    filled_layout = layout_handler.get_layout().copy(deep=True)

    history_layout = []
    history_statistics = []

    # Iterate over all entries and fill them one by one
    while not current_node.state.is_terminal():
        print(
            f"{30*'-'} Place word {current_generation+1}/{layout_handler.num_entries} {30 * '-'}"
        )
        print(f"{current_node.state}")
        current_node, statistics = searcher.search(root_node=current_node)
        print(
            f"{statistics.sort_values(by=['Reward'], ascending=False)[:10].to_string()}"
        )

        entry_before_filling = current_node.parent.state.next_entry_to_be_filled
        entry_after_filling = current_node.state.entries[entry_before_filling.index]
        chosen_word = entry_after_filling.word
        print(f"Chosen word: {chosen_word}")

        known_future_generations = searcher.get_known_depth()
        print(
            f"Known future generations: "
            f"{known_future_generations}/{layout_handler.num_entries - current_generation}"
        )

        history_statistics.append(
            {
                "index": entry_before_filling.index,
                "options": entry_before_filling.num_possible_words,
                "word": chosen_word,
                "expected_reward": statistics.loc[chosen_word, "Reward"],
                "num_visits": statistics.loc[chosen_word, "Visits"],
                "known_future_generations": known_future_generations,
            }
        )

        filled_layout = filled_layout.copy(deep=True)
        for coord, letter in zip(
            entry_after_filling.coordinates, entry_after_filling.pattern
        ):
            filled_layout.iloc[coord] = letter
        history_layout.append(filled_layout)
        print(filled_layout.to_string())

        current_generation += 1

    history_statistics_df = pd.DataFrame(history_statistics)
    print(history_statistics_df.to_string())

    final_state = current_node.state
    print(f"Final state: {final_state}")
    print(f"Final reward: {final_state.get_reward()}")

    if final_state.next_entry_to_be_filled is None:
        print("All entries have been filled successfully.")
        solved = True
    else:
        pattern = final_state.next_entry_to_be_filled.word
        print(f"Could not find word that matches pattern: {pattern}")
        solved = False

    return solved, filled_layout, history_statistics_df
