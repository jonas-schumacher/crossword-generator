from __future__ import annotations

import glob
import re
from abc import abstractmethod, ABC
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words as nltk_words

from crossword_generator.config import Constants


class WordHandler(ABC):
    def __init__(
        self,
        word_lengths: List[int],
        max_num_words: int,
    ):
        self.word_lengths = word_lengths
        self.max_num_words = max_num_words
        self.raw_words = self._read_words()
        self.clean_words = self.preprocess_words()
        self.words_by_length = self.separate_words_by_length()

    def __str__(self):
        a = f"Total number of Words: {len(self.clean_words)}\n"
        b = "\n".join(
            [f"{k}-letters: {len(v)}" for k, v in self.words_by_length.items()]
        )
        return a + b

    @abstractmethod
    def _read_words(self) -> List[str]:
        raise NotImplementedError

    def preprocess_words(self) -> List[str]:
        """
        Preprocess raw words:
        - remove non-string words and non-alphabetic characters
        - remove words that are too short or too long
        - convert to uppercase
        - remove duplicates
        - choose subset if there are too many words

        Returns
        -------
        List[str]:
            Preprocessed list of words
        """

        # Remove non-string values and words that are too short
        def filter_func(x: str | Any):
            return isinstance(x, str) and len(x) in self.word_lengths

        # Convert to uppercase and remove non-alphabetic chars
        def map_func(x: str):
            return re.sub("[^A-Z]", "", x.upper())

        transformed_words = map(
            map_func,
            filter(
                filter_func,
                self.raw_words,
            ),
        )

        # Apply transformation and remove duplicates
        words = set(transformed_words)

        # Sort words to have a unique order
        words = sorted(list(words), reverse=False)

        # Randomly chose words if word limit is exceeded
        words = list(
            np.random.choice(
                words,
                size=min(len(words), self.max_num_words),
                replace=False,
            )
        )
        return words

    def separate_words_by_length(self) -> Dict[int, List[str]]:
        """
        Separate words into lists of the same word length.

        Returns
        -------
        Dict[int, List[str]]
            int = word length
            List[str] = list of words of given word length
        """
        words_by_length = {}
        for length in self.word_lengths:
            words_by_length[length] = sorted(
                list(filter(lambda x: len(x) == length, self.clean_words))
            )
        return words_by_length


class DictionaryWordHandler(WordHandler):
    def _read_words(self) -> List[str]:
        """
        Get English words from NLTK dictionary

        Returns
        -------
        List[str]

        """
        nltk.download("words")
        words = nltk_words.words()

        return words


class FileWordHandler(WordHandler):
    def __init__(
        self,
        path_to_words: str,
        word_lengths: List[int],
        max_num_words: int,
    ):
        self.path_to_words = path_to_words
        super().__init__(
            word_lengths=word_lengths,
            max_num_words=max_num_words,
        )

    def _read_words(self) -> List[str]:
        """
        Read and merge all words from all CSV files under pattern "self.path_to_words"
        CSV files must have one column that is named "answer"

        Returns
        -------
        List[str]

        """
        paths_to_clues = [
            path
            for path in sorted(glob.glob(self.path_to_words))
            if "special" not in path
        ]

        if len(paths_to_clues) == 0:
            raise FileNotFoundError(
                f"Could not find any file with pattern {self.path_to_words}"
            )

        print(f"Use words from {len(paths_to_clues)} different files.")

        words = set.union(
            *[
                set(pd.read_csv(current_path, sep=None, engine="python")[Constants.WORD_COL_NAME].values)
                for current_path in paths_to_clues
            ]
        )

        return list(words)
