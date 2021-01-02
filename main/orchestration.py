from copy import copy
from typing import Dict, Tuple, Union

from main.data_processing import evaluate_profile
from main.manipulation import Manipulation
import numpy as np


def select_new_random_voter(failed, total_num, voter_to_exclude):
    if voter_to_exclude is not None:
        set_to_select_from = [x for x in range(total_num) if x not in failed and x != voter_to_exclude]
    else:
        set_to_select_from = [x for x in range(total_num) if x not in failed]
    if len(set_to_select_from) == 0:
        return None
    else:
        return set_to_select_from[np.random.randint(len(set_to_select_from))]


def voting_iteration(
    all_preferences, verbose, k, method, alphabetical_order, do_additions, do_omissions, do_flips, cycle_limit,
    time_limit
) -> Union[str, Tuple[bool, Dict[int, Tuple[int, int]]]]:
    """
    Full iteration per profile. 0 to many manipulations happens and ends either with convergence or not.
    Args:
        all_preferences:
        verbose:
        k:
        method:
        alphabetical_order:
        do_additions:
        do_omissions:
        do_flips:
        cycle_limit:
        time_limit:

    Returns:
    (whether_convergence, {round: (winner, voter) for all rounds}) or sring "hard_exit" if more than 30' passed
    trying to converge on this profile.
    """
    current_profile = copy(
        all_preferences
    )  # Initialize the current profile of preferences for all voters.to be the same as the truthful profile.

    num_rounds = 0
    convergence_happened = False
    res_dict = {}
    failed_manipulators = []
    manipulator_voter = None
    while True:
        random_voter = select_new_random_voter(failed_manipulators, len(all_preferences), manipulator_voter)
        if random_voter is None:
            print(f'Convergence is achieved in {num_rounds} rounds!')
            convergence_happened = True
            break
        elif verbose:
            print(f'\nRandom voter chosen: {random_voter}')
        winner, possible_winners, scores_of_alternatives = evaluate_profile(
            graphs=current_profile, k=k, method=method, alphabetical_order=alphabetical_order
        )
        if verbose:
            print(f'scores of alternatives: {scores_of_alternatives}')

        man = Manipulation(
            all_preferences=current_profile,
            preference_idx=random_voter,
            winner=winner,
            truthful_profile=all_preferences,
            possible_winners=possible_winners,
            scores_of_alternatives=scores_of_alternatives,
            alphabetical_order_of_alternatives=alphabetical_order,
            method=method,
            k=k,
            do_additions=do_additions,
            do_omissions=do_omissions,
            do_flips=do_flips,
            verbose=verbose,
            hard_exit_time_limit=time_limit
        )

        result = man.manipulation_move()
        if result == 'hard_exit':
            return 'hard_exit'

        if result is not None:
            current_profile = result[0]
            res_dict[num_rounds] = (result[1], random_voter)
            num_rounds += 1
            if verbose:
                print(f'num of round {num_rounds}')
            failed_manipulators = []
            if num_rounds > cycle_limit:
                print(f'No convergence for {cycle_limit} rounds. Assumed a cycle.')
                continue
            manipulator_voter = random_voter
        else:
            if verbose:
                print(f'Voter: {random_voter} cannot manipulate.')
            failed_manipulators.append(random_voter)

    return convergence_happened, res_dict
