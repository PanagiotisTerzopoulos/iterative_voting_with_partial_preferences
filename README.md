# Iterative voting with partial preferences - Simulation Experiments

## Introduction
This is the supplementary material for the paper "Iterative Voting with Partial Preferences" by Zoi Terzopoulou, Panagiotis Terzopoulos, and Ulle Endriss. The paper studies processes of iterative voting in settings where the agents in a group submit incomplete (specifically partial, i.e., transitive) preferences and modify them in rounds at the aim of a better outcome for themselves. Two families of voting rules (the approval family and the veto family) are analysed with respect to convergence guarantees, that is, guarantees that the iteration process will always reach stable voting states where no agent has an incentive to change her preference. 

This supplementary material includes the code required to reproduce the simulation experiments that are described in Section 5 of the paper. The experiments concern (i) the frequency of voting profiles where in the very first round of iteration no agent has an incentive to change her preference (which means that iteration never starts); (ii) the frequency of voting profiles where in the stable state the winner is the same as the one before iteration starts; (iii) the number of rounds needed for a stable state to be reached given an initial profile; (iv) the quality of the winner, in social terms, in initial profiles (before iteration) in relation to stable states (after iteration).

For every combination of voters n=10,20,50 and alternatives m=3,4,5, there are 200 complete profiles and 200 incomplete profiles generated under the IC assumption (i.e, uniformly at random) and under the 2urn model. 

## To reproduce the plots in the paper
There is a jupyter notebook called `orchestration_plots.ipynb` included in this repo which is a step-by-step guide, with ready to run code that generates a family of plots out of which specifics were chosen and presented in the paper. Feel free to try it!

## Package dependencies
Only the most basic packages are needed for one to run every part of this framework (usually get installed automatically with a full anaconda - python 3.8 installation).
Namely:
    - pandas
    - numpy
    - tqdm 
    - dill