import numpy as np
import pytest

from crossword_generator.layout_handler import NewLayoutHandler
from crossword_generator.optimizer import fill_current_layout
from crossword_generator.word_handler import DictionaryWordHandler


@pytest.mark.parametrize(
    "max_num_words, iteration_limit, solved, first_word",
    [
        (1000, 1000, True, "ASA"),
        (100, 100, False, "TAL"),
    ],
    ids=[
        "success",
        "fail",
    ],
)
def test_fill_current_layout(max_num_words, iteration_limit, solved, first_word):
    np.random.seed(123)

    layout_handler = NewLayoutHandler(
        num_rows=3,
        num_cols=3,
    )
    word_handler = DictionaryWordHandler(
        word_lengths=layout_handler.word_lengths,
        max_num_words=max_num_words,
    )

    solved_result, final_grid, final_statistics = fill_current_layout(
        layout_handler=layout_handler,
        word_handler=word_handler,
        iteration_limit=iteration_limit,
    )

    first_word_result = final_statistics.loc[0, "word"]

    # Make sure the crossword got filled when it should have
    assert solved_result == solved

    # Make sure the first word is the one we expect
    assert first_word_result == first_word
