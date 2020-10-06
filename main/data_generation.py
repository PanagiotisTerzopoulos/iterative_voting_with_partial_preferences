import math

import dill
import numpy as np
import pandas as pd

from iterative_voting.main.data_processing import check_transitivity


def fix_symmetry_diagonal(pref: array) -> array:
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


def profile_generation(alt_number: int, vot_number: int, method: str) -> LIst[np.array]:
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

    if method == 'ic':

        for x in range(0, vot_number):
            random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
            some_pref = fix_symmetry_diagonal(random_pref)
            while check_transitivity(pd.DataFrame(some_pref)) == False:
                random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
                some_pref = fix_symmetry_diagonal(random_pref)
            gprofile.append(some_pref)

    elif method == '2urn':

        random_pref_1 = np.random.randint(-1, 2, (alt_number, alt_number))
        pref_1 = fix_symmetry_diagonal(random_pref_1)
        while check_transitivity(pd.DataFrame(pref_1)) == False:
            random_pref_1 = np.random.randint(-1, 2, (alt_number, alt_number))
            pref_1 = fix_symmetry_diagonal(random_pref_1)

        random_pref_2 = np.random.randint(-1, 2, (alt_number, alt_number))
        pref_2 = fix(random_pref_2)
        while pref_1 is pref_2 or check_transitivity(pd.DataFrame(pref_2)) == False:
            random_pref_2 = np.random.randint(-1, 2, (alt_number, alt_number))
            pref_2 = fix_symmetry_diagonal(random_pref_2)

        for i in range(math.floor(vot_number / 3)):
            gprofile.append(pref_1)
            gprofile.append(pref_2)

        for i in range(vot_number - 2 * math.floor(vot_number * (1 / 3))):
            random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
            some_pref = fix_symmetry_diagonal(random_pref)
            while check_transitivity(some_pref) == False:
                random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
                some_pref = fix_symmetry_diagonal(random_pref)
            gprofile.append(some_pref)

    return gprofile


# the following is an example profile for 4 alternatives and 10 voters

one_piece_of_data = [profile_generation(4, 10, 'ic')]

### here is how to save this profile in a separate file (for Zoi's local path):

with open("C:/Users/Zoi/Documents/GitHub/iterative_voting/main/one_piece_of_data.pkl", 'wb') as f:
    dill.dump(one_piece_of_data, f)
