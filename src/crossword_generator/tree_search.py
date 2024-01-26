from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from crossword_generator.common import timing_decorator
from crossword_generator.layout_handler import LayoutHandler
from crossword_generator.state import CrosswordState


class TreeNode:
    def __init__(
        self,
        parent: Optional["TreeNode"],
        action_leading_here: Optional[str],
        state: CrosswordState,
    ):
        self.parent = parent
        self.action_leading_here = action_leading_here
        self.state = state
        self.children = {}
        self.is_terminal: bool = state.is_terminal()
        self.is_fully_expanded: bool = self.is_terminal
        self.num_visits: int = 0
        self.reward: float = 0

    def __str__(self):
        return (
            f"Terminal: {self.is_terminal}, "
            f"Expanded: {self.is_fully_expanded}, "
            f"Visits: {self.num_visits}, "
            f"Reward: {self.reward:.2f}, "
            f"Actions: {len(self.children)}"
        )


class MCTS:
    def __init__(
        self,
        layout_handler: LayoutHandler,
        iteration_limit: int,
        exploration_constant: float,
    ):
        self.root = None
        self.layout_handler = layout_handler
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant

        if iteration_limit < 1:
            raise ValueError("Iteration limit must be greater than one")

    @timing_decorator()
    def search(
        self,
        root_node: TreeNode,
    ) -> Tuple[TreeNode, pd.DataFrame]:
        """
        Explore the MCTS Tree

        Parameters
        ----------
        root_node: TreeNode
            Node from where to start the MCTS search

        Returns
        -------
        Tuple[TreeNode, pd.DataFrame]
            TreeNode: Best child node
            pd.DataFrame: Summary of search results
        """
        self.root = root_node

        # Remove the parent of the current node in order to forget everything that happened one level up
        self.root.parent = None

        # Perform as many MCTS rounds as there are iterations
        for _ in tqdm(range(self.iteration_limit)):
            self.execute_round()

        # After all MCTS rounds, get the best child of the current node
        best_child = self.get_best_child(
            parent=self.root,
            exploration_value=0.0,
        )

        # Gather some statistics about the current MCTS Tree
        statistics = pd.DataFrame(
            {
                action: {
                    "Visits": node.num_visits,
                    "Reward": node.reward,
                    "Options": node.state.num_options,
                }
                for action, node in self.root.children.items()
            }
        ).T
        return best_child, statistics

    def execute_round(self) -> None:
        """
        Execute an MCTS round consisting of
        1) selection + expansion
        3) simulation (rollout policy)
        4) backpropagation
        """
        node = self.select_node(self.root)
        reward = self.rollout_policy(node.state)
        self.backpropagate(node, reward)

    def select_node(self, node: TreeNode) -> TreeNode:
        """
        Select a node from which to start the rollout policy
        - if current node still has unknown children, select the first unknown child
        - if all children are known, follow the most promising child until an unknown descendent is found
        - if this leads to a terminal state, return that terminal state

        Parameters
        ----------
        node: TreeNode
            Root node from where to select a descendant

        Returns
        -------
        TreeNode
            Selected node from where to start the rollout policy
        """
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return self.expand(node)
            else:
                node = self.get_best_child(
                    parent=node,
                    exploration_value=self.exploration_constant,
                )
        return node

    @staticmethod
    def expand(node: TreeNode) -> TreeNode:
        """
        Add a new child to the current node with an action that has not been tried before.

        Parameters
        ----------
        node: TreeNode
            Parent node that needs to be expanded

        Returns
        -------
        TreeNode
            Child node that has just been added to parent

        """
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                new_child_node = TreeNode(
                    parent=node,
                    action_leading_here=action,
                    state=node.state.take_action(action=action),
                )
                node.children[action] = new_child_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_child_node

        raise RuntimeError(
            "This part should never be reached. "
            "It seems the node is already fully expanded even though the flag is not set."
        )

    @staticmethod
    def get_best_child(
        parent: TreeNode,
        exploration_value: float,
    ) -> TreeNode:
        """
        Get the most promising child of the current (parent) node:
        - Calculate children's value by combining their reward (exploitation) with an exploration value
        - Not explore children which are known to be a dead end
        - In case of multiple nodes with same value, take the one with a higher branching factor in the future

        Parameters
        ----------
        parent: TreeNode
            Node whose children need to be compared
        exploration_value: float
            MCTS exploration value: favors children that haven't been visited much

        Returns
        -------
        TreeNode
            Best child node

        """
        best_node = None
        best_value = float("-inf")
        best_options = float("-inf")
        for child in parent.children.values():
            node_value = child.reward + exploration_value * np.sqrt(
                np.log(parent.num_visits) / child.num_visits
            )
            node_options = child.state.num_options

            # Make sure nodes which lead to a dead end are ignored
            if node_options == 0:
                node_value = 0

            # Replace best node by current node if node_value is higher
            if node_value > best_value or (
                node_value == best_value and node_options > best_options
            ):
                best_node = child
                best_value = node_value
                best_options = node_options

        return best_node

    @staticmethod
    def rollout_policy(state: CrosswordState) -> float:
        """
        Based on a given state, take as many random actions as possible until a terminal state is reached
        Then return the reward of that final state.

        Parameters
        ----------
        state: CrosswordState
            State from which to start the rollout policy

        Returns
        -------
        float
            Reward of terminal state

        """
        while not state.is_terminal():
            try:
                action = np.random.choice(state.get_possible_actions())
            except IndexError:
                raise Exception(
                    "Non-terminal state has no possible actions: " + str(state)
                )
            state = state.take_action(action=action)
        return state.get_reward()

    @staticmethod
    def backpropagate(
        node: TreeNode,
        reward: float,
    ) -> None:
        """
        Propagate the reward of the node to all predecessors in the MCTS tree.

        Parameters
        ----------
        node: TreeNode
            Selected node from which the rollout policy started
        reward: float
            Reward received from the terminal state

        Returns
        -------
        None
        """
        while node is not None:
            node.num_visits += 1
            node.reward = max(reward, node.reward)
            node = node.parent

    def get_known_depth(self) -> int:
        """
        Calculate how many levels of the MCTS tree are known.
        1) Check if the root node knows all its children (known_generations = 1)
        2) If so, check if all children also know their children (known_generations += 1)
        3) Repeat until any node does not know all its children or if there are no children left (end of tree)

        Returns
        -------
        int:
            number of known generations below the root node
        """
        # Increase generation count by +1 for each existent and fully known child generation
        known_generations = 0
        parents = {self.root}
        while True:
            parents_fully_expanded = all([p.is_fully_expanded for p in parents])
            # Stop if some children are still unknown
            if not parents_fully_expanded:
                break
            children = set.union(*[set(p.children.values()) for p in parents])
            # Stop if current generations has no children
            if len(children) == 0:
                break
            known_generations += 1
            parents = children
        return known_generations
