import copy
from typing import List, Tuple, Union

import pandas as pd


def useful_change(parent_matrix: pd.DataFrame, index: int, type_of_alternative: str, rule: str,
                  cost: int) -> Union[List[pd.DataFrame], None]:
    """
    Gets a parent matrix and creates its child for the case of the relevant change concerning either the possible
    winner p or the current winner w.
    Args:
        parent_matrix: The matrix to change
        index: The index of the dataframe that corresponds to the row of either p or w.
        type_of_alternative: Either 'p' or 'w'
        rule: Either 'approval' or 'veto'. This is needed cause the useful change is different in its case.
        cost: Either do the one cost change or do the two cost change. Should be either 1 or 2.

    Returns:
        A list of new generated matrices or an empty list if no useful change is possible (for example when p has value
        different than -1 and the rule is 'approval')
    """
    assert type_of_alternative in ['p', 'w']
    assert rule in ['veto', 'approval']
    assert cost in [1, 2]
    new_matrices = []

    if rule == 'approval':
        if type_of_alternative == 'p':
            for col in parent_matrix.columns:
                new_matrix = copy.copy(parent_matrix)
                if parent_matrix.loc[index, col] == -1:
                    if cost == 1:
                        new_matrix.loc[index, col] = 0
                        new_matrix.loc[col, index] = 0  # symmetry constraint
                        new_matrices.append(new_matrix)
                    else:
                        new_matrix.loc[index, col] = 1
                        new_matrix.loc[col, index] = -1  # symmetry constraint
                        new_matrices.append(new_matrix)
        else:
            for col in parent_matrix.columns:
                new_matrix = copy.copy(parent_matrix)
                if (parent_matrix.loc[index, col] == 0 and
                    cost == 1) or (parent_matrix.loc[index, col] == 1 and cost == 2):
                    new_matrix.loc[index, col] = -1
                    new_matrix.loc[col, index] = 1  # symmetry constraint
                    new_matrices.append(new_matrix)

    else:
        if type_of_alternative == 'p':
            for col in parent_matrix.columns:
                new_matrix = copy.copy(parent_matrix)
                if (parent_matrix.loc[index, col] == 0 and
                    cost == 1) or (parent_matrix.loc[index, col] == -1 and cost == 2):
                    new_matrix.loc[index, col] = 1
                    new_matrix.loc[col, index] = -1  # symmetry constraint
                    new_matrices.append(new_matrix)
        else:
            for col in parent_matrix.columns:
                new_matrix = copy.copy(parent_matrix)
                if parent_matrix.loc[index, col] == 1:
                    if cost == 1:
                        new_matrix.loc[index, col] = 0
                        new_matrix.loc[col, index] = 0  # symmetry constraint
                        new_matrices.append(new_matrix)
                    else:
                        new_matrix.loc[index, col] = -1
                        new_matrix.loc[col, index] = 1  # symmetry constraint
                        new_matrices.append(new_matrix)

    return new_matrices


def one_cost_children_generation(
    parent_matrix: pd.DataFrame,
    cost_of_parent_matrix: int,
    alternatives_of_interest: List[int],
    index_of_p: int = None,
    index_of_w: int = None,
    rule: str = None,
    matrices_not_to_generate: List[pd.DataFrame] = None
) -> List[Tuple[int, list, pd.DataFrame]]:
    """
    Generates all the matrices coming of a parent matrix with cost 1.
    Args:
        parent_matrix: The matrix to generate the children matrices from.
        cost_of_parent_matrix: The cost_label of this matrix in regard with the original matrix.
        alternatives_of_interest: Needed to find the relevant cells (the only cells for which we need to examine
        what happens when their value changes). These include the possible winner, the winner, the cells that
        were relevant in the parent and for the non-transitive matrices the cells that involve an alternative
        that had to do with a change also become relevant.
        index_of_p: If the index of possible winner is provided only the useful changes will be done for this row.
        index_of_w: If the index of the winner is provided only the useful changes will be done for this row.
        rule: Either "approval" or "veto". Only relevant if index_of_p or index_of_w is provided.
        matrices_not_to_generate: Skip these matrices. Useful to avoid generate matrices that had been generated
        somewhere else in the tree.

    Returns:
        A list of tuples: (cost-label_of_child, indices_changed_from_the_parent, child)
    """
    # TODO: make this function more efficient. We shouldn't generate all the matrices everytime and in the end drop
    #  the "matrices_not_generate", but generate only the ones we need from the beginning. The ones we don't need are
    #  some that change specific cells that had be changed in the past (not to be confused with relevant_cells which
    #  are actually relevant rows). So we need an argument that passes these specific cells ((row, col) tuple) and we
    #  don't need at all the "matrices_not_to_generate" in this case.
    if index_of_w or index_of_p:
        assert rule in ['veto', 'approval']

    relevant_rows = parent_matrix.index.isin(alternatives_of_interest)
    children_matrices = []
    for rel, row in zip(relevant_rows, parent_matrix.index):
        for col in parent_matrix.columns:
            if rel and row != col:
                if row == index_of_p:
                    new_matrices = useful_change(
                        parent_matrix=parent_matrix, index=index_of_p, type_of_alternative='p', rule=rule, cost=1
                    )
                    if new_matrices:
                        children_matrices += [
                            (cost_of_parent_matrix + 1, [index_of_w], new_matrix) for new_matrix in new_matrices
                        ]
                elif row == index_of_w:
                    new_matrices = useful_change(
                        parent_matrix=parent_matrix, index=index_of_w, type_of_alternative='w', rule=rule, cost=1
                    )
                    if new_matrices:
                        children_matrices += [
                            (cost_of_parent_matrix + 1, [index_of_w], new_matrix) for new_matrix in new_matrices
                        ]
                else:
                    # do additions
                    if parent_matrix.loc[row, col] == 0:
                        new_matrix = copy.copy(parent_matrix)
                        new_matrix.loc[row, col] = 1
                        new_matrix.loc[col, row] = -1  # symmetry constraint
                        children_matrices.append((cost_of_parent_matrix + 1, [row], new_matrix))

                        new_matrix = copy.copy(parent_matrix)
                        new_matrix.loc[row, col] = -1
                        new_matrix.loc[col, row] = 1  # symmetry constraint
                        children_matrices.append((cost_of_parent_matrix + 1, [row], new_matrix))
                    # do ommits
                    if parent_matrix.loc[row, col] == 1 or parent_matrix.loc[row, col] == -1:
                        new_matrix = copy.copy(parent_matrix)
                        new_matrix.loc[row, col] = 0
                        new_matrix.loc[col, row] = 0  # symmetry constraint
                        children_matrices.append((cost_of_parent_matrix + 1, [row], new_matrix))

    if matrices_not_to_generate:
        return [x for x in children_matrices if not True in [x[2].equals(y) for y in matrices_not_to_generate]]
    else:
        return children_matrices


def two_cost_children_generation(
    parent_matrix: pd.DataFrame,
    cost_of_parent_matrix: int,
    alternatives_of_interest: List[int],
    index_of_p: int = None,
    index_of_w: int = None,
    rule: str = None,
    matrices_not_to_generate: List[pd.DataFrame] = None
) -> List[Tuple[int, list, pd.DataFrame]]:
    """
    Generates all the matrices coming of a parent matrix with cost 2.
    Args:
        parent_matrix: The matrix to generate the children matrices from.
        cost_of_parent_matrix: The cost_label of this matrix in regard with the original matrix.
        alternatives_of_interest: Needed to find the relevant cells (the only cells for which we need to examine
        what happens when their value changes). These include the possible winner, the winner, the cells that
        were relevant in the parent and for the non-transitive matrices the cells that involve an alternative
        that had to do with a change also become relevant.
        index_of_p: If the index of possible winner is provided only the useful changes will be done for this row.
        index_of_w: If the index of the winner is provided only the useful changes will be done for this row.
        rule: Either "approval" or "veto". Only relevant if index_of_p or index_of_w is provided.
        matrices_not_to_generate: Skip these matrices. Useful to avoid generate matrices that had been generated
        somewhere else in the tree.

    Returns:
        A list of tuples: (cost-label_of_child, child)
    """
    # TODO: make this function more efficient. We shouldn't generate all the matrices everytime and in the end drop
    #  the "matrices_not_generate", but generate only the ones we need from the beginning. The ones we don't need are
    #  some that change specific cells that had be changed in the past (not to be confused with relevant_cells which
    #  are actually relevant rows). So we need an argument that passes these specific cells ((row, col) tuple) and we
    #  don't need at all the "matrices_not_to_generate" in this case.
    if index_of_w or index_of_p:
        assert rule in ['veto', 'approval']

    relevant_rows = parent_matrix.index.isin(alternatives_of_interest)
    children_matrices = []
    for rel, row in zip(relevant_rows, parent_matrix.index):
        for col in parent_matrix.columns:
            if rel and row != col:
                if row == index_of_p:
                    new_matrices = useful_change(
                        parent_matrix=parent_matrix, index=index_of_p, type_of_alternative='p', rule=rule, cost=2
                    )
                    if new_matrices:
                        children_matrices += [
                            (cost_of_parent_matrix + 2, [index_of_p], new_matrix) for new_matrix in new_matrices
                        ]
                elif row == index_of_w:
                    new_matrices = useful_change(
                        parent_matrix=parent_matrix, index=index_of_w, type_of_alternative='w', rule=rule, cost=2
                    )
                    if new_matrices:
                        children_matrices += [
                            (cost_of_parent_matrix + 2, [index_of_w], new_matrix) for new_matrix in new_matrices
                        ]
                else:
                    # do flips
                    if parent_matrix.loc[row, col] == 1 or parent_matrix.loc[row, col] == -1:
                        new_matrix = copy.copy(parent_matrix)
                        new_matrix.loc[row, col] = parent_matrix.loc[col, row]
                        new_matrix.loc[col, row] = parent_matrix.loc[row, col]  # symmetry constraint
                        children_matrices.append((cost_of_parent_matrix + 2, [row], new_matrix))

    if matrices_not_to_generate:
        return [x for x in children_matrices if not True in [x[2].equals(y) for y in matrices_not_to_generate]]
    else:
        return children_matrices


def find_matrices_with_score(matrices: List[Tuple[int, list, pd.DataFrame]],
                             score: int) -> List[Tuple[int, list, pd.DataFrame]]:
    """
    Gets a list of preferences with their absolute scores and returns a subset of this list with the matrices having
    only the score provided.
    Args:
        matrices: All the input matrices. A superset of the returned.
        score: Returns matrices with this score.
    """
    return [x for x in matrices if x[0] == score]
