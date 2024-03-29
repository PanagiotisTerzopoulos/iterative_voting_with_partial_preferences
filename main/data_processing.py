"""
This module includes utility functions that have to do with calculating basic preferences and profile attributes like
the score of a specific alternative in a preference or the winner given a profile.
"""

from typing import List, Tuple

import pandas as pd


def get_score_of_alternative_by_voter(graph: pd.DataFrame, method: str, k: int, alternative: int) -> int:
    if method == 'approval':
        try:
            if graph.iloc[alternative].value_counts()[-1.0] < k:
                return 1
        except KeyError:
            return 1
        return 0
    elif method == 'veto':
        try:
            if graph.iloc[alternative].value_counts()[1.0] > k - 1:
                return 1
        except KeyError:
            return 0
        return 0


def find_sum_of_alternatives(graphs: List[pd.DataFrame], k: int, method: str, num_of_alternatives: int) -> dict:
    assert method in ['approval', 'veto']

    d = {}
    for alternative in range(num_of_alternatives):
        d[str(alternative)] = 0
        for g in graphs:
            d[str(alternative)] += get_score_of_alternative_by_voter(g, method, k, alternative)

    return d


def get_winners_from_scores(
    scores_of_alternatives: dict,
    alphabetical_order: dict,
) -> Tuple[int, List[int]]:
    max_score = max(scores_of_alternatives.values())
    winners = [int(key) for key, value in scores_of_alternatives.items() if value == max_score]
    winner_key = int(min([key for key, value in alphabetical_order.items() if value in winners]))
    winner = alphabetical_order[winner_key]
    possible_winners = [int(key) for key, value in scores_of_alternatives.items() if value >= max_score - 2]
    possible_winners.remove(winner)
    return winner, possible_winners


def evaluate_profile(graphs: List[pd.DataFrame], k: int, method: str,
                     alphabetical_order: dict) -> Tuple[int, List[int], dict]:
    num_of_alternatives = len(graphs[0])
    scores_of_alternatives = find_sum_of_alternatives(graphs, k, method, num_of_alternatives)
    winner, possible_winners = get_winners_from_scores(scores_of_alternatives, alphabetical_order)
    return winner, possible_winners, scores_of_alternatives


def check_transitivity(graph: pd.DataFrame) -> bool:
    aces_series = graph.apply(lambda row: row[row == 1].index.tolist(), axis=1)
    for i, row in enumerate(aces_series):
        for ace_column in row:
            for j in aces_series.loc[ace_column]:
                if graph.loc[i, j] != 1:
                    return False
    return True


def get_social_welfare_of_alternative(profile: List[pd.DataFrame], alt: int):
    '''
    This function is only neeeded for the orchestration_plots.ipynb notebook that reproduces the plots of the
    original publication.
    '''
    social_welfare = 0
    for vot in profile:
        social_welfare += ((vot.loc[alt] == 1).sum() - (vot.loc[alt] == -1).sum())
    return social_welfare
