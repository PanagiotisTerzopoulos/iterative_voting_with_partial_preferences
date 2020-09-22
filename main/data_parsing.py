# WARNING: these work only for truncated preferances
from typing import List, Tuple

import numpy as np
import pandas as pd

from .data_processing import check_transitivity


def parse_file(
    fpath: str
) -> Tuple[list, int, int]:
    with open(fpath) as f:
        lines = f.readlines()
    num_of_alternatives = int(lines[0])
    num_voters, _, num_unique_prefs = lines[num_of_alternatives + 1].split(',')
    num_voters = int(num_voters)

    prefs = lines[(num_of_alternatives + 2):]
    profile = []
    for p in prefs:
        num_of_times = int(p.split(',')[0])
        pref = p.split(',')[1:]
        pref = [int(x) for x in pref]
        for i in range(num_of_times):
            profile.append(pref)

    return profile, num_of_alternatives, num_voters


def generate_graphs_from_profile(
    profile: list,
    num_of_alternatives: int
) -> List[pd.DataFrame]:
    graphs = []
    for pref_num, pref in enumerate(profile):
        adj_matrix = np.zeros((num_of_alternatives, num_of_alternatives))
        non_existing = [x for x in range(1, num_of_alternatives + 1) if x not in pref]
        for i, vote in enumerate(pref):
            for j in (pref[i + 1:] + non_existing):
                if j != vote:
                    adj_matrix[vote - 1][j - 1] = 1
                    adj_matrix[j - 1][vote - 1] = -1

        ids = [x for x in range(adj_matrix.shape[0])]
        graphs.append(pd.DataFrame(adj_matrix, index=ids, columns=ids))
    return graphs


def create_random_data(num_of_voters: int, num_alternatives: int, uniqueness: bool = False) -> List[pd.DataFrame]:
    """
    Creates a random list of preferences in the format of adjacency matrices with the only condition of them
    satisfying transitivity (and ofc being symmetric). Optionally they can also be unique.
    """
    mats = []
    while len(mats) < num_of_voters:
        new_mat = np.random.randint(-1, 2, (num_alternatives, num_alternatives), dtype='int8')
        np.fill_diagonal(new_mat, 0)
        new_mat = pd.DataFrame(new_mat)
        if check_transitivity(new_mat):
            if not uniqueness:
                mats.append(pd.DataFrame(np.tril(new_mat.values) + np.tril(new_mat.values, -1).T))
            else:
                no_duplicate = True
                for m in mats:
                    if m.equals(new_mat):
                        no_duplicate = False
                        print('duplicate')
                if no_duplicate:
                    mats.append(pd.DataFrame(np.tril(new_mat.values) + np.tril(new_mat.values, -1).T))
    return mats
