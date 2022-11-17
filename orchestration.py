"""
This is the main entrypoint script which will automatically run all the computational experiments on the complete or
the incomplete data, generate and save the results. See the bottom for all the possible arguments that can be passed
along the script call.

E.g. of script call:
```
 python orchestration.py --num_alt 5 --num_voter 10 --data_type 2urn --k 4 --method approval --time_limit 900
 ```
"""

import argparse
import os

import dill
from tqdm import tqdm

from main.orchestration import voting_iteration


def main(args):
    if args.complete_profiles:
        suffix = '_complete'
    else:
        suffix = '_incomplete'
    with open(f'data/our_data{suffix}.pkl', 'rb') as f:
        all_data = dill.load(f)

    alphabetical_order = {}
    for i in range(args.num_alt):
        alphabetical_order[i] = i

    data_to_use = all_data[(args.num_voters, args.num_alt, args.data_type)]
    print(f'running {args}')

    if os.path.isfile('data/results/total_result.pkl'):
        with open('data/results/total_result.pkl', 'rb') as f:
            total_result = dill.load(f)
    else:
        total_result = {}

    prof_indices_to_run = list(range(len(data_to_use))) if args.random_choice is None else [args.random_choice]

    for random_profile in tqdm(prof_indices_to_run, desc='random profiles'):
        all_preferences = data_to_use[random_profile]
        for meta_counter in range(args.num_iterations):
            key = (
                args.num_alt, args.num_voters, args.data_type, random_profile, args.k, args.method, meta_counter,
                args.do_additions, args.do_omissions, args.do_flips, args.complete_profiles
            )
            calculate_it = False
            if key not in total_result.keys():
                calculate_it = True
            elif total_result[key] == 'hard_exit' and args.retry_slow_ones:
                calculate_it = True
            elif args.overwrite:
                calculate_it = True

            if calculate_it:
                result = voting_iteration(
                    all_preferences, args.verbose, args.k, args.method, alphabetical_order, args.do_additions,
                    args.do_omissions, args.do_flips, args.time_limit
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_alt', type=int, default=3)
    parser.add_argument('--num_voters', type=int, default=10)
    parser.add_argument('--data_type', type=str, default='ic')
    parser.add_argument('--random_choice', type=int, default=None)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--method', type=str, default='approval')
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--do_additions', type=bool, default=True)
    parser.add_argument('--do_omissions', type=bool, default=True)
    parser.add_argument('--do_flips', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--retry_slow_ones', type=bool, default=False)
    parser.add_argument('--time_limit', type=int, default=900)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--complete_profiles', type=bool, default=False)

    args = parser.parse_args()
    assert args.k <= args.num_alt

    main(args)
