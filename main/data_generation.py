"""
Module that includes all utilities functions that have to do with the artificial data generation.
"""

import os
import random
import sys
from typing import List

import numpy as np
import pandas as pd

# The small patch below sovles importing modules issues on windows computers.
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


def profile_generation(alt_number: int, vot_number: int, method: str, complete: bool) -> List[pd.DataFrame]:
    """
        Inputs:
         alt_number: the number of alternatives
         vot_number: the number of voters
         method: the way that we want the profile to be created. e.g., what kind of distribution to sample from.
          we include "ic" for impartial culture when we create the preference uniformly at random and
          "2urn" for the 2urn model (see literature for details on the definition of this)
          complete: whether to generate complete or incomplete preferences
        Returns:
        A profile: a list of arrays
    """
    if complete:
        func = generate_complete_random_preference
    else:
        func = generate_incomplete_random_preference

    gprofile = []

    assert method in ['ic', '2urn']

    if method == 'ic':
        for _ in range(vot_number):
            some_pref = func(alt_number)
            gprofile.append(some_pref)
    else:
        pref_1 = func(alt_number)
        pref_2 = func(alt_number)
        while pref_2.equals(pref_1):
            pref_2 = func(alt_number)

        for _ in range(vot_number):
            random_pref_assignment = random.randrange(3)
            if random_pref_assignment == 0:
                gprofile.append(pref_1)
            elif random_pref_assignment == 1:
                gprofile.append(pref_2)
            else:
                pref_3 = func(alt_number)
                while pref_3.equals(pref_1) or pref_3.equals(pref_2):
                    pref_3 = func(alt_number)
                gprofile.append(pref_3)

    return gprofile


def generate_incomplete_random_preference(alt_number: int) -> pd.DataFrame:
    random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
    some_pref = pd.DataFrame(fix_symmetry_diagonal(random_pref))
    while not check_transitivity(some_pref):
        random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
        some_pref = pd.DataFrame(fix_symmetry_diagonal(random_pref))
    return some_pref


def generate_complete_random_preference(alt_number: int) -> pd.DataFrame:
    order = list(range(alt_number))
    random.shuffle(order)
    new_mat = np.ones((alt_number, alt_number))
    for counter, alt in enumerate(order):
        for i in range(0, counter):
            new_mat[alt, order[i]] = -1
    np.fill_diagonal(new_mat, 0)
    return pd.DataFrame(new_mat).astype(int)
