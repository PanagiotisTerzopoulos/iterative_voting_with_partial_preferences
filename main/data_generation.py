import math
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

curr_path = os.path.realpath(__file__)
if curr_path.split('/iterative_voting/main')[0] not in sys.path:
    sys.path.append(curr_path.split('/iterative_voting/main')[0])

from iterative_voting.main.data_processing import check_transitivity


def fix_symmetry_diagonal(pref: np.array) -> np.array:
    """
        Takes as input a random array and modifies it so that it is symmetric
        (actually symmetric cells should have opposite values) and is has zeros in the diagonal.
        Returns:
        The modified array
    """
    for i in range(0, len(pref)):
        for j in range(0, len(pref)):
            if j > i:
                pref[i][j] = -pref[j][i]
            elif j == i:
                pref[i][j] = 0
    return pref


def profile_generation(alt_number: int, vot_number: int, method: str) -> List[pd.DataFrame]:
    """
        Inputs:
         alt_number: the number of alternatives
         vot_number: the number of voters
         method: the way that we want the profile to be created. e.g., what kind of distribution to sample from.
          we include "ic" for impartial culture when we create the preference uniformly at random and
          "2urn" for the 2urn model (see literature for details on the definition of this)
        Returns:
        A profile: a list of arrays
    """
    gprofile = []

    assert method in ['ic', '2urn']

    if method == 'ic':
        for _ in range(vot_number):
            some_pref = generate_random_preference(alt_number)
            gprofile.append(some_pref)
    else:
        pref_1 = generate_random_preference(alt_number)
        pref_2 = generate_random_preference(alt_number)
        while pref_2.equals(pref_1):
            pref_2 = generate_random_preference(alt_number)

        for _ in range(math.floor(vot_number / 3)):
            gprofile.append(pref_1)
            gprofile.append(pref_2)

        for i in range(vot_number - 2 * math.floor(vot_number / 3)):
            gprofile.append(generate_random_preference(alt_number))

    # TODO: create 3urn method as well

    return gprofile


def generate_random_preference(alt_number: int):
    random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
    some_pref = pd.DataFrame(fix_symmetry_diagonal(random_pref))
    while not check_transitivity(some_pref):
        random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
        some_pref = pd.DataFrame(fix_symmetry_diagonal(random_pref))
    return some_pref
