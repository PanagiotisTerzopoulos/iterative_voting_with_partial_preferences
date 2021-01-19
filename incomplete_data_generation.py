import os

import dill
from tqdm import tqdm

from main.data_generation import profile_generation

num_voters = [10, 20, 50]
num_alternatives = [3, 4, 5]
methods = ['2urn']

comb = []
for v in num_voters:
    for alt in num_alternatives:
        for m in methods:
            comb.append((v, alt, m))

if os.path.isfile('data/our_data_incomplete.pkl'):
    with open('data/our_data_incomplete.pkl', 'rb') as f:
        profiles = dill.load(f)
else:
    profiles = {}

for c in tqdm(comb):
    print(c)
    profiles[c] = {
        i: profile_generation(c[1], c[0], c[2], complete=False) for i in tqdm(range(200), desc='random iterations')
    }
with open('data/our_data_incomplete.pkl', 'wb') as f:
    dill.dump(profiles, f)
