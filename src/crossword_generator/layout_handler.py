from typing import Tuple, List

import numpy as np
import pandas as pd

from crossword_generator.common import Orientation
from crossword_generator.config import (
    Constants,
    DefaultArguments,
)


def get_slice(
    row: int,
    col: int,
    orientation: Orientation,
) -> Tuple[slice, slice]:
    if orientation == Orientation.ACROSS:
        return slice(row, row + 1), slice(col, None)
    elif orientation == Orientation.DOWN:
        return slice(row, None), slice(col, col + 1)
    else:
        raise ValueError(f"Unknown orientation type {orientation}")


def get_coordinates(
    row: int,
    col: int,
    orientation: Orientation,
    length: int,
) -> List[Tuple[int, int]]:
    if orientation == Orientation.ACROSS:
        return [(row, col + k) for k in range(length)]
    elif orientation == Orientation.DOWN:
        return [(row + k, col) for k in range(length)]
    else:
        raise ValueError(f"Unknown orientation type {orientation}")


class LayoutHandler:
    def __init__(
        self,
        initial_layout: pd.DataFrame,
    ):
        self.layout = initial_layout
        self.num_rows = self.layout.shape[0]
        self.num_cols = self.layout.shape[1]
        self.num_cells: int = self.num_rows * self.num_cols
        self.num_blocks: int = (self.layout == Constants.BLOCK_SYMBOL).sum().sum()
        self.share_of_blocks: float = self.num_blocks / self.num_cells

        self.patterns, self.coordinates = self.extract_entries_from_layout(
            layout=self.layout
        )
        self.dependencies = self.get_dependencies(coordinates=self.coordinates)
        self.word_start_grid = self.get_word_position_grid(
            num_rows=self.num_rows, num_cols=self.num_cols, coordinates=self.coordinates
        )
        self.num_entries: int = len(self.coordinates)
        self.entries: List[int] = [i for i in range(self.num_entries)]

        self.entry_lengths = [len(entry) for entry in self.coordinates]

        self.word_lengths = sorted(list(set(self.entry_lengths)))

    def __str__(self):
        return (
            f"{self.word_start_grid.to_string()}\n"
            f"{self.layout.to_string()}\n"
            f"Number of cells: {self.num_cells} "
            f"(Regular: {self.num_cells-self.num_blocks} + Blocks: {self.num_blocks})\n"
            f"Number of Entries: {self.num_entries}\n"
            f"Word lengths: {self.word_lengths}"
        )

    def get_layout(self) -> pd.DataFrame:
        return self.layout

    @staticmethod
    def extract_entries_from_layout(
        layout: pd.DataFrame,
    ) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
        """
        Given a layout, extract a list of entries (= gaps to be filled with words).
        Logic is as follows: a new word has to start
        - right to each block and in the first column if there is no block
        - below each block and in the first row if there is not a block already

        If the given layout potential words shorter than MIN_WORD_LENGTH they are treated as if they were separated

        Returns
        -------
        List[List[Tuple[int, int]]]
            - List No1 = Entries
            - List No2 = Coordinates
            - Tuple = Row and Column
        """
        # Temporarily replace block symbol:
        layout = layout.copy(deep=True).replace(
            Constants.BLOCK_SYMBOL, Constants.BLOCK_SEPARATOR
        )
        num_rows = layout.shape[0]
        num_cols = layout.shape[1]
        coordinates = []
        patterns = []
        for row in range(num_rows):
            for col in range(num_cols):
                orientations_to_add = []
                # 1: Skip blocks
                if layout.iloc[row, col] == Constants.BLOCK_SEPARATOR:
                    continue

                # 2: Insert a horizontal word in each row of the first column or if there is a block to the left:
                if col == 0 or layout.iloc[row, col - 1] == Constants.BLOCK_SEPARATOR:
                    orientations_to_add.append(Orientation.ACROSS)

                # 3: Insert a vertical word in each column of the first row or if there is a block above:
                if row == 0 or layout.iloc[row - 1, col] == Constants.BLOCK_SEPARATOR:
                    orientations_to_add.append(Orientation.DOWN)

                for orientation in orientations_to_add:
                    word = "".join(
                        layout.iloc[
                            get_slice(
                                row=row,
                                col=col,
                                orientation=orientation,
                            )
                        ].values.flatten()
                    )
                    # If there is a block in the potential word, take the part before the next block
                    if Constants.BLOCK_SEPARATOR in word:
                        word = word.split(Constants.BLOCK_SEPARATOR)[0]
                    # If the word is too short, do not create an entry for it
                    if len(word) < DefaultArguments.MIN_WORD_LENGTH:
                        continue
                    patterns.append(list(word))
                    coordinates.append(
                        get_coordinates(
                            row=row,
                            col=col,
                            orientation=orientation,
                            length=len(word),
                        )
                    )

        return patterns, coordinates

    @staticmethod
    def get_dependencies(
        coordinates: List[List[Tuple[int, int]]],
    ) -> List[List[Tuple[int, int]]]:
        """
        For each coordinate of each entry, mark which other entry is affected by it

        Parameters
        ----------
        coordinates: List[List[Tuple[int, int]]]
            Coordinates of all entries

        Returns
        -------
        List[List[int]]
            Dependencies of all entries

        """
        dependencies = []
        for pA, pA_coordinates in enumerate(coordinates):
            dependencies.append([])
            for pA_coord in pA_coordinates:
                for pB, pB_coordinates in enumerate(coordinates):
                    if pA == pB:
                        continue
                    for pB_pos, pB_coord in enumerate(pB_coordinates):
                        if pA_coord == pB_coord:
                            dependencies[pA].append((pB, pB_pos))
        return dependencies

    @staticmethod
    def get_word_position_grid(
        num_rows: int,
        num_cols: int,
        coordinates: List[List[Tuple[int, int]]],
    ) -> pd.DataFrame:
        """
        Create a DataFrame that shows all coordinates of all entries

        Parameters
        ----------
        num_rows: int
        num_cols: int
        coordinates: List[List[Tuple[int, int]]]

        Returns
        -------
        pd.DataFrame(num_rows, num_cols)
        """
        word_position_grid = pd.DataFrame(
            [[[] for col in range(num_cols)] for row in range(num_rows)]
        )
        for p, p_coordinates in enumerate(coordinates):
            for coord in p_coordinates:
                word_position_grid.iloc[
                    coord[Constants.ROW_INDEX], coord[Constants.COL_INDEX]
                ].append(p)
        return word_position_grid

    def add_blocks_to_layout(
        self,
        num_blocks: int,
    ) -> "LayoutHandler":
        new_layout = self._add_blocks_to_given_layout(
            layout=self.layout,
            num_blocks=num_blocks,
        )
        return LayoutHandler(initial_layout=new_layout)

    @staticmethod
    def _add_blocks_to_given_layout(
        layout: pd.DataFrame,
        num_blocks: int,
    ) -> pd.DataFrame:
        """
        Add a certain number of blocks to the given layout.
        Add one block after the other at a random position.
        Stop when placing another block would mean creating entries that are too short

        Parameters
        ----------
        layout: pd.DataFrame
        num_blocks: int

        Returns
        -------
        pd.DataFrame

        """
        layout = layout.replace(Constants.BLOCK_SYMBOL, Constants.BLOCK_SEPARATOR)
        num_rows = layout.shape[0]
        num_cols = layout.shape[1]
        neighbors = {
            "above": {
                "row": lambda row: slice(0, row),
                "col": lambda col: slice(col, col + 1),
                "index": -1,
            },
            "below": {
                "row": lambda row: slice(row + 1, num_rows),
                "col": lambda col: slice(col, col + 1),
                "index": 0,
            },
            "left": {
                "row": lambda row: slice(row, row + 1),
                "col": lambda col: slice(0, col),
                "index": -1,
            },
            "right": {
                "row": lambda row: slice(row, row + 1),
                "col": lambda col: slice(col + 1, num_cols),
                "index": 0,
            },
        }
        blocks_filled = 0
        while blocks_filled < num_blocks:
            possible_coordinates = []
            for row in range(num_rows):
                for col in range(num_cols):
                    # If there is a block already, the coordinate should be skipped:
                    if layout.iloc[row, col] == Constants.BLOCK_SEPARATOR:
                        continue
                    possible = True
                    for k, v in neighbors.items():
                        word = "".join(
                            layout.iloc[v["row"](row), v["col"](col)].values.flatten()
                        )
                        if Constants.BLOCK_SEPARATOR in word:
                            word = word.split(Constants.BLOCK_SEPARATOR)[v["index"]]
                        if len(word) in list(
                            range(1, DefaultArguments.MIN_WORD_LENGTH)
                        ):
                            possible = False
                            break
                    if possible:
                        possible_coordinates.append((row, col))
            if len(possible_coordinates) > 0:
                remove_row, remove_col = possible_coordinates[
                    np.random.choice(len(possible_coordinates))
                ]
                layout.iloc[remove_row, remove_col] = Constants.BLOCK_SEPARATOR
                blocks_filled += 1
                if DefaultArguments.SYMMETRIC_DESIGN:
                    layout.iloc[
                        num_rows - remove_row - 1, num_cols - remove_col - 1
                    ] = Constants.BLOCK_SEPARATOR
                    blocks_filled += 1
            else:
                print("Unable to add next block to current layout.")
                break
        layout = layout.replace(Constants.BLOCK_SEPARATOR, Constants.BLOCK_SYMBOL)
        return layout


class NewLayoutHandler(LayoutHandler):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
    ):
        layout = self._create_new_layout(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        super().__init__(initial_layout=layout)

    @staticmethod
    def _create_new_layout(
        num_rows: int,
        num_cols: int,
    ) -> pd.DataFrame:
        """
        Create a new layout without blocks based on height and width

        Parameters
        ----------
        num_rows: int
        num_cols: int

        Returns
        -------
        pd.DataFrame

        """
        layout = pd.DataFrame(
            data=Constants.EMPTY_SYMBOL,
            index=range(num_rows),
            columns=range(num_cols),
        )

        return layout


class ExistingLayoutHandler(LayoutHandler):
    def __init__(
        self,
        path_to_existing_layout: str,
    ):
        layout = self._read_existing_layout(
            path_to_existing_layout=path_to_existing_layout,
        )
        super().__init__(initial_layout=layout)

    @staticmethod
    def _read_existing_layout(
        path_to_existing_layout: str = None,
    ) -> pd.DataFrame:
        """
        Read an existing layout from given path.
        The given layout must be a CSV file with
        - arbitrary index and columns
        - coordinates that are blocks must be empty
        - regular coordinates may contain any symbol

        Parameters
        ----------
        path_to_existing_layout: str

        Returns
        -------
        pd.DataFrame

        """
        layout = pd.read_csv(
            path_to_existing_layout,
            index_col=0,
            keep_default_na=False,
        )

        return layout
