"""
This script is meant to be run individually to generate artificial data of complete preferences to run experiments on.
"""

import os

import dill
from tqdm import tqdm

from main.data_generation import profile_generation
'''
The parameters below can be changed. Data will be generated for all the combinations of these.
'''
num_voters = [10, 20, 50]
num_alternatives = [3, 4, 5]
methods = ['2urn']

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
    print(c)
    profiles[c] = {
        i: profile_generation(c[1], c[0], c[2], complete=True) for i in tqdm(range(200), desc='random iterations')
    }
with open('data/our_data_complete.pkl', 'wb') as f:
    dill.dump(profiles, f)
