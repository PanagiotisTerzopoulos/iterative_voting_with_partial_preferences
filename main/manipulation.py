import copy
import os
import random
import sys
from typing import List, Tuple, Union

import pandas as pd

curr_path = os.path.realpath(__file__)
if curr_path.split('/iterative_voting/main')[0] not in sys.path:
    sys.path.append(curr_path.split('/iterative_voting/main')[0])

from iterative_voting.main.manipulation_utils import find_matrices_with_score, get_children_generation_options, \
    one_cost_children_generation, \
    two_cost_children_generation
from .data_processing import check_transitivity, evaluate_profile, get_score_of_alternative_by_voter, \
    get_winners_from_scores


class Manipulation:
    """
    Check your voter's preference. is there a possible winner (p) that the voter truthfully ranks above winner(w)? If
    no, pick next voter. If yes, check (from the most preferred such p to the least preferred one) whether the voter
    can manipulate:
         - if the voter gives p score 1 and w score 0, not possible.
         - if the voter gives p score 0 and w score 0, check whether p would win with score 1. If no, move on. If yes,
          find the minimum number of edges to add/omit/flip (respecting transitivity and the relevant costs) to give p
          score 1 (randomly pick if there is more than one way). Choose a new random agent and go back to
           step 2.
         - if the voter gives p score 1 and w score 1, check whether p would win if w has score 0. If no, move one. If
         yes, find the minimum number of edges to add/omit/flip (respecting transitivity and the relevant costs) to give
          w score 0 (randomly pick if there is more than one way). Choose a new random agent and go back
          to step 2.
         - if the voter gives p score 0 and w gets score 1, check as above: increasing p, lowering w, or both.
    """

    def __init__(
        self,
        all_preferences: List[pd.DataFrame],
        preference_idx: int,  # The index for the 'all_preferences' list that corresponds to the preference of the
        # specific voter.
        truthful_profile: List[pd.DataFrame],
        absolute_cost_of_preference: int,
        winner: int,
        possible_winners: List[int],
        scores_of_alternatives: dict,
        alphabetical_order_of_alternatives: dict,
        method: str,
        k: int,
        do_additions: bool,
        do_omissions: bool,
        do_flips: bool
    ):

        self.all_preferences = all_preferences  # These are the preferences of all the voters in a list,
        # in the beginning of the current manipulation round.
        self.preference_idx = preference_idx
        self.preference = all_preferences[preference_idx]  # The preference of the specific voter.
        self.truthful_profile = truthful_profile  # This is the original profile, before any manipulation of any voter.
        self.absolute_cost_of_preference = absolute_cost_of_preference
        self.winner = winner
        self.possible_winners = possible_winners
        self.scores_of_alternatives = scores_of_alternatives  # The total scores of the alternatives in the profile,
        # not the voter specific scores.
        self.method = method
        self.k = k
        self.alphabetical_order_of_alternatives = alphabetical_order_of_alternatives
        self.all_generated_matrices: List[Tuple[int, list, pd.DataFrame]] = [(0, [], self.preference)]  # All the
        # matrices
        # generated as children while exploring the tree of possible manipulations + the original matrix (
        # preference). (cost-label_of_child, indices_changed_from_the_parent, child)
        self.do_additions = do_additions
        self.do_omissions = do_omissions
        self.do_flips = do_flips

    def check_for_possible_manipulation(self) -> bool:
        """
        Check our voter's preference to see if there is a possible winner (p) that the voter truthfully ranks
        above w.
        Returns:
            Whether there is at least one such p.
        """
        for p in self.possible_winners:
            if self.preference.loc[p, self.winner] == 1:
                return True
        return False

    def get_alternatives_order(self, alternatives: List[int]) -> List[int]:
        """
        Gets a list of alternatives existing in a preference and returns the same list sorted from the most preferred
        alternative to the least preferred one according to the preference of the voter.
        """
        sorted_alternatives = []
        pref = self.truthful_profile[self.preference_idx].loc[alternatives, alternatives]
        while True:
            top_ones = pref[pref.apply(lambda row: -1 not in row.values, axis=1)].index.values
            if top_ones.size > 0:
                sorted_alternatives.append(top_ones[0])
                pref = pref.drop(top_ones[0]).drop(columns=[top_ones[0]])
            else:
                sorted_alternatives += pref.index.values.tolist()
                break

        assert set(alternatives) == set(sorted_alternatives)
        return sorted_alternatives

    def manipulation_move(self) -> Union[None, Tuple[List[pd.DataFrame], int]]:
        """
        Main functionality which checks all the scenaria of possible manipulation of the result by the specific voter
        and if the voter manages to manipulate it returns the voter's updated preference, otherwise returns None.

        Returns:
            None if Manipulation cannot happen or the new profile and the the winner in a tuple if manipulation happens.
        """
        # alternatives_to_check are all the possible winners that the current voter truthfully prefers to the profile
        # winner (self.w)
        alternatives_to_check = [
            x for x in self.possible_winners if self.truthful_profile[self.preference_idx].loc[x, self.winner] == 1
        ]
        sorted_alternatives = self.get_alternatives_order(alternatives_to_check)
        winner_score = get_score_of_alternative_by_voter(self.preference, self.method, self.k, self.winner)
        for p in sorted_alternatives:
            p_score = get_score_of_alternative_by_voter(self.preference, self.method, self.k, p)

            if p_score == 1 and winner_score == 0:
                continue

            if p_score == 0 and winner_score == 0:
                scores_of_alternatives = copy.deepcopy(self.scores_of_alternatives)
                # check whether p would win with score 1
                scores_of_alternatives[str(p)] += 1
                would_win = get_winners_from_scores(scores_of_alternatives,
                                                    self.alphabetical_order_of_alternatives)[0] == p
                potential_winners = []
                if would_win:
                    all_prefs, manipulation_happened, winner = self.tree_generation(
                        p, potential_winners=potential_winners
                    )

                    if manipulation_happened:
                        return all_prefs, winner
                else:
                    continue

            if p_score == 1 and winner_score == 1:
                # check whether p would win if winner had score 0
                scores_of_alternatives = copy.deepcopy(self.scores_of_alternatives)
                scores_of_alternatives[str(self.winner)] -= 1
                potential_winners = [
                    get_winners_from_scores(scores_of_alternatives, self.alphabetical_order_of_alternatives)[0]
                ]  # These possible_winners are not the same as above.
                continue_with_next_alternative = False
                while potential_winners[-1] != p:
                    if get_score_of_alternative_by_voter(
                        self.preference, self.method, self.k, potential_winners[-1]
                    ) == 0:
                        continue_with_next_alternative = True
                        break
                    else:
                        # check whether p would win if possible_winner (say x) had score 0
                        scores_of_alternatives[str(potential_winners[-1])] -= 1
                        new_potential_winner = get_winners_from_scores(
                            scores_of_alternatives, self.alphabetical_order_of_alternatives
                        )[0]
                        if new_potential_winner == potential_winners[-1]:
                            # no new winners are found
                            continue_with_next_alternative = True
                            break
                        else:
                            potential_winners.append(new_potential_winner)

                if continue_with_next_alternative:
                    continue
                else:
                    '''
                    that means that possible_winner == p. the alternatives of interest here are all possible_winners:
                    "note that the relevant cells, besides the alternative w, will also concern all other 
                    alternatives that have to have their scores lowered"
                    '''
                    all_prefs, manipulation_happened, winner = self.tree_generation(
                        p, potential_winners=potential_winners
                    )
                    if manipulation_happened:
                        return all_prefs, winner
                    pass

            if p_score == 0 and winner_score == 1:
                # The logic of this section is the same as above
                # check whether p would win if winner had score 0

                # # TODO: this is wrong. It could be either of the two or both or in other order... Thus, we 'll skip it
                # #  for now (it only added efficiency anyway)
                # scores_of_alternatives[str(self.winner)] -= 1
                # scores_of_alternatives[str(p)] += 1
                # potential_winners = [
                #     get_winners_from_scores(scores_of_alternatives, self.alphabetical_order_of_alternatives)[0]
                # ]
                #
                # continue_with_next_alternative = False
                # while potential_winners[-1] != p:
                #     if get_score_of_alternative_by_voter(
                #         self.preference, self.method, self.k, potential_winners[-1]
                #     ) == 0:
                #         continue_with_next_alternative = True
                #         break
                #     else:
                #         # check whether p would win if possible_winner (say x) had score 0
                #         scores_of_alternatives[str(potential_winners[-1])] -= 1
                #         new_potential_winner = get_winners_from_scores(
                #             scores_of_alternatives, self.alphabetical_order_of_alternatives
                #         )[0]
                #         if new_potential_winner == potential_winners[-1]:
                #             # no new winners are found
                #             continue_with_next_alternative = True
                #             break
                #         else:
                #             potential_winners.append(new_potential_winner)
                #
                # if continue_with_next_alternative:
                #     continue
                # else:
                #     '''
                #     that means that possible_winner == p. the alternatives of interest here are all possible_winners:
                #     "note that the relevant cells, besides the alternative w, will also concern all other
                #     alternatives that have to have their scores lowered"
                #     '''
                #     all_prefs, manipulation_happened, winner = self.tree_generation(
                #         p, potential_winners=potential_winners
                #     )
                #     if manipulation_happened:
                #         return all_prefs, winner
                #     pass

                potential_winners = [int(x) for x in self.scores_of_alternatives.keys()]
                all_prefs, manipulation_happened, winner = self.tree_generation(p, potential_winners=potential_winners)

                if manipulation_happened:
                    return all_prefs, winner

        return None

    def tree_generation(self, p, potential_winners: list = None):
        """
        Given an alternative p that is a possible winner and is preferred by our voter to the current winner, we want
        to check whether the voter can make p win. In order to do this, we generate matrices that differ from the
        original matrix (i.e., the current preference of the voter) in rounds:
        (the intuition we have is a tree-shape, where each level consists of matrices of the same cost).
        we start from those that have cost 1,
        we continue to those that have cost 2, and so on. After we check all matrices that have the same cost and we
        know that none of them is transitive and makes p win, we procced to the next round and generate matrices that
        have further differences with the original one. The key idea is that we don't need to make all possible changes,
        but only those that are "useful"---those that concern the current winner and the possible winner p (and
        potentially some other alternatives that need to have their scores changed too). We distinguish between
        transitive matrices, in which only useful changes need to be considered in future levels of the tree,
        and non-transitive matrices, in which some more alternatives become relevant because changes that concern them
        may induce a transitive matrix that makes p win in following levels.

        Returns:
        ???

        """
        print(f'Alternative {p} is preferred. Investigating possible manipulation.')
        all_prefs = copy.deepcopy(self.all_preferences)
        # first generate all the 1-cost children of the original matrix
        new_preferences = one_cost_children_generation(
            parent_matrix=self.preference,
            cost_of_parent_matrix=self.absolute_cost_of_preference,
            alternatives_of_interest=list(set([p, self.winner] + potential_winners)),
            index_of_p=p,
            index_of_w=self.winner,
            rule=self.method,
            do_additions=self.do_additions,
            do_omissions=self.do_omissions
        )
        self.all_generated_matrices += new_preferences
        all_prefs, manipulation_happened, winner = self.check_if_manipulation_happened(all_prefs, new_preferences, p)
        if not manipulation_happened:
            # since the direct one-cost children of the original didn't work, for every child generate the
            # 1-cost children and the 2-cost children from the previous matrix.
            all_prefs, manipulation_happened, winner = self.tree_generation_level_1_onwards(
                all_prefs, p, potential_winners
            )
        return all_prefs, manipulation_happened, winner

    def tree_generation_level_1_onwards(self, all_prefs: List[pd.DataFrame], p: int, potential_winners: list):
        """

        """
        print('in tree_generation_level_1_onwards')
        while True:
            if self.all_generated_matrices:
                old_max_cost_so_far = max([x[0] for x in self.all_generated_matrices])
            else:
                old_max_cost_so_far = 0
            matrices_to_examine_cost_1 = find_matrices_with_score(self.all_generated_matrices, old_max_cost_so_far)
            matrices_to_examine_cost_2 = find_matrices_with_score(self.all_generated_matrices, old_max_cost_so_far - 1)

            func_to_use_first = {
                0: (self.examine_matrices_cost_2, matrices_to_examine_cost_2),
                1: (self.examine_matrices_cost_1, matrices_to_examine_cost_1)
            }
            order = random.randint(0, 1)
            all_prefs, manipulation_happened, new_preferences, winner = func_to_use_first[order][0](
                all_prefs, func_to_use_first[order][1], old_max_cost_so_far, p, potential_winners
            )
            if manipulation_happened:
                break
            order = 0 if order else 1
            all_prefs, manipulation_happened, new_preferences, winner = func_to_use_first[order][0](
                all_prefs, func_to_use_first[order][1], old_max_cost_so_far, p, potential_winners
            )
            if manipulation_happened:
                break

            self.all_generated_matrices += new_preferences
            # We exit the while loop  naturally when all relevant cells (resulting from each relevant
            # alternatives) have been changed and no manipulation happened
            if self.all_generated_matrices:
                new_max_cost_so_far = max([x[0] for x in self.all_generated_matrices])
            else:
                new_max_cost_so_far = 0
            if new_max_cost_so_far == old_max_cost_so_far:
                break
        return all_prefs, manipulation_happened, winner

    def examine_matrices_cost_2(self, all_prefs, matrices_to_examine_cost_2, old_max_cost_so_far, p, potential_winners):
        new_preferences = []
        for parent_mat_2 in matrices_to_examine_cost_2:
            index_of_p, index_of_w, relevant_cells = get_children_generation_options(self.winner, p, parent_mat_2)
            new_preferences += two_cost_children_generation(
                parent_matrix=parent_mat_2[2],
                cost_of_parent_matrix=old_max_cost_so_far - 1,
                alternatives_of_interest=list(set(relevant_cells + potential_winners)),
                index_of_p=index_of_p,
                index_of_w=index_of_w,
                rule=self.method,
                matrices_not_to_generate=[x[2] for x in self.all_generated_matrices],
                do_flips=self.do_flips
            )
        all_prefs, manipulation_happened, winner = self.check_if_manipulation_happened(all_prefs, new_preferences, p)
        return all_prefs, manipulation_happened, new_preferences, winner

    def examine_matrices_cost_1(self, all_prefs, matrices_to_examine_cost_1, old_max_cost_so_far, p, potential_winners):
        new_preferences = []
        for parent_mat_1 in matrices_to_examine_cost_1:
            index_of_p, index_of_w, relevant_cells = get_children_generation_options(self.winner, p, parent_mat_1)
            new_preferences += one_cost_children_generation(
                parent_matrix=parent_mat_1[2],
                cost_of_parent_matrix=old_max_cost_so_far,
                alternatives_of_interest=list(set(relevant_cells + potential_winners)),
                index_of_p=index_of_p,
                index_of_w=index_of_w,
                rule=self.method,
                matrices_not_to_generate=[x[2] for x in self.all_generated_matrices],
                do_omissions=self.do_omissions,
                do_additions=self.do_additions
            )
        all_prefs, manipulation_happened, winner = self.check_if_manipulation_happened(all_prefs, new_preferences, p)
        return all_prefs, manipulation_happened, new_preferences, winner

    def check_if_manipulation_happened(
        self, all_prefs: List[pd.DataFrame], new_preferences: List[Tuple[int, list, pd.DataFrame]], p: int
    ) -> Tuple[List[pd.DataFrame], bool, int]:
        """

        """
        manipulation_happened = False
        winner = None
        for pref_cost, _, pref in new_preferences:
            all_prefs[self.preference_idx] = pref
            # TODO: NB: make this code more efficient. we don't need to evaluate the whole profile every
            #  time. we have the scores of alternatives dictionary and we can calculate with the
            #  'get_score_of_alternative_by_voter' function the score the old preference gives and the
            #  one the new pref gives. Substract once for all the alternatives the old scores of the
            #  voter and then add the new ones (for all alternatives) each time you check a preference.
            winner, *_ = evaluate_profile(all_prefs, self.k, self.method, self.alphabetical_order_of_alternatives)
            if winner == p and check_transitivity(pref):
                print('Manipulation happened!')
                manipulation_happened = True
                break
        return all_prefs, manipulation_happened, winner
