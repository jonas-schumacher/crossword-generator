from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from crossword_generator import generate_crossword
from crossword_generator.config import DefaultArguments


def parse_arguments() -> Namespace:
    argument_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument(
        "--path_to_layout",
        type=str,
        default=DefaultArguments.PATH_TO_LAYOUT,
    )
    argument_parser.add_argument(
        "--num_rows",
        type=int,
        default=DefaultArguments.NUM_ROWS,
    )
    argument_parser.add_argument(
        "--num_cols",
        type=int,
        default=DefaultArguments.NUM_COLS,
    )
    argument_parser.add_argument(
        "--path_to_words",
        type=str,
        default=DefaultArguments.PATH_TO_WORDS,
    )
    argument_parser.add_argument(
        "--max_num_words",
        type=int,
        default=DefaultArguments.MAX_NUM_WORDS,
    )
    argument_parser.add_argument(
        "--max_mcts_iterations",
        type=int,
        default=DefaultArguments.MAX_MCTS_ITERATIONS,
    )
    argument_parser.add_argument(
        "--random_seed",
        type=int,
        default=DefaultArguments.RANDOM_SEED,
    )
    argument_parser.add_argument(
        "--output_path",
        type=str,
        default=DefaultArguments.OUTPUT_PATH,
    )
    arguments = argument_parser.parse_args()
    return arguments


def main() -> None:
    arguments = parse_arguments()
    generate_crossword(
        path_to_layout=arguments.path_to_layout,
        num_rows=arguments.num_rows,
        num_cols=arguments.num_cols,
        path_to_words=arguments.path_to_words,
        max_num_words=arguments.max_num_words,
        max_mcts_iterations=arguments.max_mcts_iterations,
        random_seed=arguments.random_seed,
        output_path=arguments.output_path,
    )


if __name__ == "__main__":
    main()
