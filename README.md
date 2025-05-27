# MCTS Crossword Generator

This package provides a pure Python implementation for generating crosswords using
Monte Carlo Tree Search (MCTS).
- A general overview about the project can be found in this [blog post](http://schumacher.pythonanywhere.com/homepage/crossword).
- If you want to understand the MCTS implementation in more detail, I recommend this [code deep dive on Medium](https://medium.com/@jonas.schumacher/generate-your-own-crosswords-using-single-player-mcts-3bf6bceeea40).
- The pip package can be found on [PyPI](https://pypi.org/project/crossword-generator)

![screenshot filled crossword](https://github.com/jonas-schumacher/crossword-generator/raw/main/images/layout_5_12_empty.png)
![screenshot filled crossword](https://github.com/jonas-schumacher/crossword-generator/raw/main/images/layout_5_12_filled.png)

## Quickstart

### A: Install package:
1. Create and activate a virtual environment based on Python >= 3.8 
2. Install crossword_generator package: 
```
pip install crossword-generator
```

### B: Generate crossword with default settings:
You can generate a crossword without providing any arguments.
This will fill a 4x5 layout without any black squares using words from an English dictionary. 

To do so, activate your virtual environment and chose one of the following (equivalent) options:
1. Call application directly:
```
crossword
```
2. Execute package: 
```
python -m crossword_generator
```
3. Run the main function in a python shell or your own script:
```
>>> from crossword_generator import generate_crossword
>>> generate_crossword()
````

For the next examples I assume you are using the first option to interact with the package.

## Examples
- To get started and see which input formats are required, you can download some
[english](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_en)
(comma-separated) or
[german](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_de)
(semicolon-separated) sample data.
- Let's assume you have downloaded the sample files into a directory called "crossword_input" inside your working directory.

### A: Use your own layouts

- In order to use your own layouts, you will need to set argument 
`path_to_layout` to a CSV file on your local machine. 
- the CSV file must have an index column and a header row
- potential letters are marked with "_" (underscore)
- black squares are marked with "" (empty)

Fill an 
[empty 5x12 layout](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_en/layout_5_12_empty.csv):
```
crossword --path_to_layout "crossword_input/layout_5_12_empty.csv"
```

Fill a 
[prefilled 5x12 layout](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_en/layout_5_12_prefilled.csv):
```
crossword --path_to_layout "crossword_input/layout_5_12_prefilled.csv"
```

Fill an entire NYT-style
[15x15 layout](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_en/layout_15_15_empty.csv):

```
crossword --path_to_layout "crossword_input/layout_15_15_empty.csv"
```

Of course, you can also provide arguments from within your code:
```
generate_crossword(
  path_to_layout="crossword_input/layout_15_15_empty.csv",
)
```

### B: Use your own words

- In order to use your own set of words, you will need to set argument 
`path_to_words` to a CSV file (or pattern of CSV files) on your local machine.
- the CSV file(s) must contain a column named "answer" with the relevant words

Fill an 
[empty 5x12 layout](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_en/layout_5_12_empty.csv)
with [words from a list](https://github.com/jonas-schumacher/crossword-generator/tree/main/sample_input_en/sample_words.csv)
```
crossword --path_to_layout "crossword_input/layout_5_12_empty.csv" --path_to_words "crossword_input/sample_words.csv"
```

Again, you can do the same from within your code:
```
generate_crossword(
  path_to_layout="crossword_input/layout_5_12_empty.csv",
  path_to_words="crossword_input/sample_words.csv",
)
```


### C: Other arguments you might want to play with:
- *num_rows & num_cols [int]*
    - number of rows / columns the layout should have
    - will only be considered if `path_to_layout` is not specified
- *max_num_words [int]*
    - limits the number of words to improve runtime
- *max_mcts_iterations [int]*
    - sets the maximum number of MCTS iterations
    - can be increased to get a better solution or decreased to improve runtime
- *random_seed [int]*
    - change the seed to obtain different filled crosswords
- *output_path [str]*
    - if provided, save the final grid and a summary as CSV files into the provided directory

## Modules
- **optimizer.py**
  - script that contains the main function `generate_crossword()`
- **layout_handler.py**
  - Provides the layout that will later be filled with words
  - `NewLayoutHandler`: creates a new layout from scratch 
  - `ExistingLayoutHandler`: reads an existing layout from a CSV file
- **word_handler.py**
  - Provides the words that will later be filled into the layout
  - `DictionaryWordHandler`: get words from NLTK corpus
  - `FileWordHandler`: read words from CSV files
- **state.py**
  - `Entry`: class that represents the current state of one entry of the crossword
  - `CrosswordState`: class that represents the current state of whole crossword
- **tree_search.py**
  - `TreeNode`: represents one node of the MCTS tree
  - `MCTS`: represents the whole MCTS tree and provides all necessary functionalities such as
    - Selection
    - Expansion
    - Simulation / Rollout
    - Backpropagation

## References & Dependencies
- The MCTS implementation in `tree_search.py` is based on the algorithm provided by [pbsinclair42](https://github.com/pbsinclair42/MCTS),
   which I adapted in several ways:
  - Convert from 2-player to 1-player domain
  - Adjust reward function + exploration term
  - Add additional methods to analyze the game tree
  - Use PEP 8 code style
- Have a look at `pyproject.toml` for a list of all required and optional dependencies
- Python >= 3.8
- Required packages
  - nltk>=3.5
  - pandas>=1.4.0
  - numpy>=1.22.0
  - tqdm>=4.41.0

## Future work
- Add a python module that creates questions for given answers using NLP techniques
- Add a graphical user interface (GUI)