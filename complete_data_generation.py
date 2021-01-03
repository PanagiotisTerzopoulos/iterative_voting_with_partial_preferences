import os

import dill
from tqdm import tqdm

from main.data_generation import profile_generation

num_voters = [10, 20, 50, 70]
num_alternatives = [3, 4, 5, 6, 7]
methods = ['ic', '2urn']

comb = []
for v in num_voters:
    for alt in num_alternatives:
        for m in methods:
            comb.append((v, alt, m))

if os.path.isfile('data/our_data_complete.pkl'):
    with open('data/our_data_complete.pkl', 'rb') as f:
        profiles = dill.load(f)
else:
    profiles = {}

for c in tqdm(comb):
    if c not in profiles.keys():
        print(c)
        profiles[c] = {
            i: profile_generation(c[1], c[0], c[2], complete=True) for i in tqdm(range(200), desc='random iterations')
        }
        with open('data/our_data_complete.pkl', 'wb') as f:
            dill.dump(profiles, f)
