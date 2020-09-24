import numpy as np
import copy
import dill


def transitive(pref):
    trans = True
    for k in range(0, len(pref)):
        for l in range(0, len(pref)):
            for r in range(0, len(pref)):
                if pref[k][l] == 1 and pref[l][r] == 1:
                    if pref[k][r] != 1:
                        trans = False
    return trans

def fix(pref):
    fixed_pref = pref
    for i in range(0, len(pref)):
        for j in range(0, len(pref)):
            if j > i:
                fixed_pref[i][j] = -pref[j][i]
            elif j == i:
                fixed_pref[i][j] = 0
    return fixed_pref

def profile_generation(m,n, method):
    vot_number = n
    alt_number = m

    if method == 'ic':
        gprofile = []
        for x in range(0, vot_number):
            random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
            some_pref = fix(random_pref)
            while transitive(some_pref) == False:
                random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
                some_pref = fix(random_pref)
            gprofile.append(some_pref)

    elif method == '2urn':
        gprofile = []

        random_pref_1 = np.random.randint(-1, 2, (alt_number, alt_number))
        pref_1 = fix(random_pref_1)
        while transitive(pref_1) == False:
            random_pref_1 = np.random.randint(-1, 2, (alt_number, alt_number))
            pref_1 = fix(random_pref_1)

        random_pref_2 = np.random.randint(-1, 2, (alt_number, alt_number))
        pref_2 = fix(random_pref_2)
        while pref_1 is pref_2 or transitive(pref_2) == False:
            random_pref_2 = np.random.randint(-1, 2, (alt_number, alt_number))
            pref_2 = fix(random_pref_2)

        for i in range (math.floor(vot_number/3)):
            gprofile.append(pref_1)
            gprofile.append(pref_2)

        for i in range (vot_number-2*math.floor(vot_number*(1/3))):
            random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
            some_pref = fix(random_pref)
            while transitive(some_pref) == False:
                random_pref = np.random.randint(-1, 2, (alt_number, alt_number))
                some_pref = fix(random_pref)
            gprofile.append(some_pref)

    return gprofile

my_data = []

for k in range(100):
    my_data.append(profile_generation(4,10, 'ic'))

with open("C:/Users/Zoi/Documents/GitHub/iterative_voting/main/my_uniform_data.pkl", 'wb') as f:
    dill.dump(my_data, f)






