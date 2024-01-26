from __future__ import annotations

import re
from copy import deepcopy
from typing import List, Tuple

from crossword_generator.common import convert_pattern_for_matching
from crossword_generator.config import Constants
from crossword_generator.layout_handler import LayoutHandler
from crossword_generator.word_handler import WordHandler


class Entry:
    def __init__(
        self,
        index: int,
        length: int,
        coordinates: List[Tuple[int, int]],
        dependencies: List[Tuple[int, int]],
        pattern: List[str],
        possible_words: List[str],
        num_letters_fixed: int,
        word_fixed: bool,
    ) -> None:
        self.index: int = index
        self.length: int = length
        self.coordinates: List[Tuple[int, int]] = coordinates
        self.dependencies: List[Tuple[int, int]] = dependencies

        self.pattern: List[str] = pattern
        self.possible_words: List[str] = possible_words
        self.num_letters_fixed: int = num_letters_fixed
        self.word_fixed: bool = word_fixed

    def __str__(self):
        return (
            f"Entry: #{self.index}, "
            f"Position: ({self.coordinates[0][Constants.ROW_INDEX]}, {self.coordinates[0][Constants.COL_INDEX]}), "
            f"Word: {self.word}({self.num_letters_fixed}/{self.length}), "
            f"Possible words: {self.num_possible_words}."
        )

    @property
    def word(self):
        return "".join(self.pattern)

    @property
    def num_possible_words(self):
        return len(self.possible_words)


class CrosswordState:
    """
    Class representing a crossword board
    """

    def __init__(
        self,
        entries: List[Entry],
    ) -> None:
        self.entries: List[Entry] = entries
        self.empty_entries: List[Entry] = [p for p in entries if not p.word_fixed]
        self.filled_entries: List[Entry] = [p for p in entries if p.word_fixed]
        self.words_already_used: List[str] = [p.word for p in self.filled_entries]
        self.next_entry_to_be_filled: Entry | None = self.get_next_entry()

    def __str__(self):
        return (
            f"State: {len(self.filled_entries)}/{len(self.entries)}, "
            f"{self.next_entry_to_be_filled}"
        )

    @property
    def next_index(self) -> int | None:
        return (
            self.next_entry_to_be_filled.index
            if self.next_entry_to_be_filled is not None
            else None
        )

    @property
    def num_options(self) -> int | None:
        return (
            self.next_entry_to_be_filled.num_possible_words
            if self.next_entry_to_be_filled is not None
            else None
        )

    def get_next_entry(self) -> Entry | None:
        """
        Get the next entry by choosing the one with the smallest number of possible words

        Returns
        -------
        Entry, if there are still empty entries
        None, if terminal state is reached and all entries are filled
        """
        return (
            min(self.empty_entries, key=lambda x: x.num_possible_words)
            if len(self.empty_entries) != 0
            else None
        )

    def get_possible_actions(self) -> List[str]:
        possible_actions = self.next_entry_to_be_filled.possible_words
        return possible_actions

    def take_action(
        self,
        action: str,
    ) -> "CrosswordState":
        """
        Calculate the next state after taking an 'action', to do so
        1) Copy all entries from the current state that have not been affected by the action
        2) Fill the entry that is represented by the taken action
        3) For all entries that share a letter with the action:
        - change their pattern at the relevant position
        - recalculate the possible words

        Parameters
        ----------
        action: str

        Returns
        -------
        CrosswordState
            Resulting state after taking action

        """
        # Create a shallow copy of all existing entries
        new_entries = [old for old in self.entries]

        # Overwrite the next entry (the one that was just fixed by the action taken)
        next_old = self.next_entry_to_be_filled
        next_new = Entry(
            index=next_old.index,
            length=next_old.length,
            coordinates=next_old.coordinates,
            dependencies=next_old.dependencies,
            pattern=list(action),
            possible_words=[action],
            num_letters_fixed=next_old.length,
            word_fixed=True,
        )
        new_entries[next_old.index] = next_new

        for current_position, (
            affected_entry_index,
            affected_entry_position,
        ) in enumerate(next_old.dependencies):
            affected_old = self.entries[affected_entry_index]

            # If the affected entry is already fixed, there is no need to update it
            if affected_old.word_fixed:
                continue

            # Overwrite pattern at relevant position
            pattern = deepcopy(affected_old.pattern)
            pattern[affected_entry_position] = next_new.pattern[current_position]

            # Reduce possible words to those that match the pattern
            pattern_for_matching = convert_pattern_for_matching(pattern)
            possible_words = [
                word
                for word in affected_old.possible_words
                if re.fullmatch(pattern_for_matching, word)
                and word not in self.words_already_used + [action]
            ]
            affected_new = Entry(
                index=affected_old.index,
                length=affected_old.length,
                coordinates=affected_old.coordinates,
                dependencies=affected_old.dependencies,
                pattern=pattern,
                possible_words=possible_words,
                num_letters_fixed=affected_old.num_letters_fixed + 1,
                word_fixed=False,
            )
            new_entries[affected_old.index] = affected_new

        next_state = CrosswordState(entries=new_entries)

        return next_state

    def is_terminal(self) -> bool:
        """
        Calculate if this state is a terminal state which can happen in two cases:
        A: The state is a success, if all entries have been filled
        B: The state is a fail, if at least one entry is unable to get filled

        Returns
        -------
        bool:
            True if state is terminal
        """
        # If there is no empty entry, all words have been filled successfully
        success = len(self.empty_entries) == 0
        # If there still is a next entry, but not word can
        fail = (
            self.next_entry_to_be_filled is not None
            and self.next_entry_to_be_filled.num_possible_words == 0
        )
        return success or fail

    def get_reward(self) -> float:
        """
        Calculate reward by simply checking how many entries have been filled successfully

        Returns
        -------
        float
        """
        reward = len(self.filled_entries)
        return reward


def get_initial_crossword_state(
    layout_handler: LayoutHandler,
    word_handler: WordHandler,
) -> CrosswordState:
    """
    Create a new CrosswordState object by creating empty entries as defined in LayoutHandler

    Parameters
    ----------
    layout_handler: LayoutHandler
    word_handler: WordHandler

    Returns
    -------
    CrosswordState
        Initial (empty) crossword state
    """
    entries = []
    for i in layout_handler.entries:
        length = layout_handler.entry_lengths[i]
        num_letters_fixed = len(
            list(
                filter(
                    lambda x: x != Constants.EMPTY_SYMBOL, layout_handler.patterns[i]
                )
            )
        )
        word_fixed = num_letters_fixed == length
        pattern = layout_handler.patterns[i]
        possible_words = word_handler.words_by_length[length]
        if word_fixed:
            possible_words = ["".join(pattern)]
        elif num_letters_fixed > 0:
            pattern_for_matching = convert_pattern_for_matching(pattern)
            possible_words = [
                word
                for word in possible_words
                if re.fullmatch(pattern_for_matching, word)
            ]
        entries.append(
            Entry(
                index=i,
                length=length,
                coordinates=layout_handler.coordinates[i],
                dependencies=layout_handler.dependencies[i],
                pattern=layout_handler.patterns[i],
                possible_words=possible_words,
                num_letters_fixed=num_letters_fixed,
                word_fixed=word_fixed,
            )
        )
    return CrosswordState(entries=entries)
