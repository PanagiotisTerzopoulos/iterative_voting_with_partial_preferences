import argparse
import os

import dill
from tqdm import tqdm

from main.orchestration import voting_iteration


def main():
    for num_alt in [3, 4, 5]:
        for num_voters in [10, 20, 50]:
            for complete_profiles in [True, False]:
                for data_type in ['ic', '2urn']:
                    method = 'approval'
                    time_limit = 900
                    retry_slow_ones = False
                    k = 1
                    cycle_limit = 100
                    do_omissions = True
                    do_additions = True
                    do_flips = True
                    random_choice = None
                    num_iterations = 50
                    overwrite = False
                    verbose = False

                    if complete_profiles:
                        suffix = '_complete'
                    else:
                        suffix = '_incomplete'
                    with open(f'data/our_data{suffix}.pkl', 'rb') as f:
                        all_data = dill.load(f)
                
                    alphabetical_order = {}
                    for i in range(num_alt):
                        alphabetical_order[i] = i
                
                    data_to_use = all_data[(num_voters, num_alt, data_type)]

                
                    if os.path.isfile('data/results/total_result.pkl'):
                        with open('data/results/total_result.pkl', 'rb') as f:
                            total_result = dill.load(f)
                    else:
                        total_result = {}
                
                    prof_indices_to_run = list(range(len(data_to_use))) if random_choice is None else [random_choice]
                
                    for random_profile in tqdm(prof_indices_to_run, desc='random profiles'):
                        all_preferences = data_to_use[random_profile]
                        for meta_counter in range(num_iterations):
                            key = (
                                num_alt, num_voters, data_type, random_profile, k, method, cycle_limit,
                                meta_counter, do_additions, do_omissions, do_flips, complete_profiles
                            )

                            calculate_it = False
                            if key not in total_result.keys():
                                calculate_it = True
                            elif total_result[key] == 'hard_exit' and retry_slow_ones:
                                calculate_it = True
                            elif overwrite:
                                calculate_it = True
                
                            if calculate_it:
                                print(f'running {key}')
                                result = voting_iteration(
                                    all_preferences, verbose, k, method, alphabetical_order, do_additions,
                                    do_omissions, do_flips, cycle_limit, time_limit
                                )
                                if result == 'hard_exit':
                                    total_result[key] = 'hard_exit'
                                    with open('data/results/total_result.pkl', 'wb') as f:
                                        dill.dump(total_result, f)
                                    break
                                else:
                                    convergence_happened, res_dict = result
                                    total_result[key] = (convergence_happened, res_dict)
                                    # if it cannot manipulate for this profile then it doesn't make sense running the profile
                                    # multiple times cause the random order doesn't play a role
                                    with open('data/results/total_result.pkl', 'wb') as f:
                                        dill.dump(total_result, f)
                                    if not res_dict:
                                        break


if __name__ == '__main__':
    main()
