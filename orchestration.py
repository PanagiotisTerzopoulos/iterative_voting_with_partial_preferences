from copy import copy

import dill
import numpy as np
from main.data_processing import evaluate_profile
from main.manipulation import Manipulation


def boolean_input(x):
    if x == "True":
        return True
    elif x == 'False':
        return False
    else:
        raise ValueError


def select_new_random_voter(failed, total_num, voter_to_exclude):
    if voter_to_exclude is not None:
        set_to_select_from = [x for x in range(total_num) if x not in failed and x != voter_to_exclude]
    else:
        set_to_select_from = [x for x in range(total_num) if x not in failed]
    if len(set_to_select_from) == 0:
        return None
    else:
        return set_to_select_from[np.random.randint(len(set_to_select_from))]


data_inp = input('Manual example to load?')
with open(f'data/profile_manual_example_{data_inp}.pkl', 'rb') as f:
    all_preferences = dill.load(f)

alphabetical_order = {}
for i in all_preferences[0].index:
    alphabetical_order[i] = i

k = int(input('k?'))
assert k <= len(all_preferences[0]), 'k can be at most the number of alternatives'
method = input('method? (approval/method)')
assert method in ['veto', 'approval']
cycle_limit = int(input('Cycle limit?') or 100)
num_iterations = int(input('Num iterations?') or 5)
do_additions = boolean_input(input('do additions?') or 'True')
do_omissions = boolean_input(input('do omissions?') or 'True')
do_flips = boolean_input(input('do flips?') or 'True')
verbose = boolean_input(input('verbose?') or 'False')

convergence_rounds = []
for meta_counter in range(num_iterations):

    current_profile = copy(
        all_preferences
    )  # Initialize the current profile of preferences for all voters.to be the same as the truthful profile.
    num_rounds = 0
    failed_manipulators = []
    manipulator_voter = None

    while True:
        random_voter = select_new_random_voter(failed_manipulators, len(all_preferences), manipulator_voter)
        if random_voter is None:
            print(f'Convergence is achieved in {num_rounds} rounds!')
            convergence_rounds.append(num_rounds)
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
            verbose=verbose
        )

        result = man.manipulation_move()
        if result is not None:
            current_profile, _ = result
            num_rounds += 1
            failed_manipulators = []
            if num_rounds > cycle_limit:
                print(f'No convergence for {cycle_limit} rounds. Assumed a cycle.')
                continue
            manipulator_voter = random_voter
        else:
            if verbose:
                print(f'Voter: {random_voter} cannot manipulate.')
            failed_manipulators.append(random_voter)
