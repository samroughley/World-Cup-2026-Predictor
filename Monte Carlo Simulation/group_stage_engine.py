import pandas as pd 
import json 
from collections import defaultdict


## Load data ##

# Elo data
with open('../team_info/initial_elo_ratings.json', 'r') as f:
    elo_ratings = json.load(f)


def calculate_outcome_probability(team_A, team_B, outcome):
    """
    Returns an estimate for the outcome probabilities of a match.
    """

    # Could add a check to see if it has happened here, alllowing 
    # probabilities to update as the compeition progresses

    # Get the elo scores
    elo_a = elo_ratings[team_A]
    elo_b = elo_ratings[team_B]

    # Calculate the two scores
    Q_a = 10 ** (elo_a/400)
    Q_b = 10 ** (elo_b/400)

    # Calculate the two probabilities
    p_a = Q_a / (Q_a + Q_b)
    p_b = 1 - p_a

    # Add in a guess for a draw probability
    p_diff = abs(p_a-p_b)
    draw_prob = (1-p_diff)/2  # linear scaling
    p_a *= (1-draw_prob)
    p_b *= (1-draw_prob)

    if outcome == "H":
        return p_a
    elif outcome == "A":
        return p_b 
    elif outcome == "D":
        return draw_prob
    else:
        "Another outcome?"

    

def break_ties(teams):
    """
    Split cases where two teams have the same number of points.

    Break ties randomly equally for now, later substitute for a better
    model.
    """

    import math 
    import itertools

    num_sequences = math.factorial(len(teams))
    return {tuple(t_seq): 1/num_sequences for t_seq in itertools.permutations(teams)}



def simulate_group(teams):
    """
    Return all possible numbers of points for the teams,
    with the respective probabilities.

    Uses calculate_outcome_probability to determine the probabilities
    for individual match outcomes.
    """

    team_1, team_2, team_3, team_4 = teams
    outcome_probs = {}

    for match_1_outcome in ["H","D","A"]:   # 1 vs 2
        for match_2_outcome in ["H","D","A"]:   # 3 vs 4
            for match_3_outcome in ["H","D","A"]:   # 1 vs 3
                for match_4_outcome in ["H","D","A"]:   # 2 vs 4
                    for match_5_outcome in ["H","D","A"]:   # 1 vs 4
                        for match_6_outcome in ["H","D","A"]:   # 2 vs 3

                            # Initialise table and probability
                            team_1_points, team_2_points, team_3_points, team_4_points = 0, 0, 0, 0
                            prob = 1

                            ## Construct table and overall probability ##

                            # Match 1
                            if match_1_outcome == "H":
                                team_1_points += 3
                            elif match_1_outcome == "D":
                                team_1_points += 1
                                team_2_points += 1
                            else:
                                team_2_points += 3
                            prob *= calculate_outcome_probability(team_1, team_2, match_1_outcome)

                            # Match 2
                            if match_2_outcome == "H":
                                team_3_points += 3
                            elif match_2_outcome == "D":
                                team_3_points += 1
                                team_4_points += 1
                            else:
                                team_4_points += 3
                            prob *= calculate_outcome_probability(team_3, team_4, match_2_outcome)

                            # Match 3
                            if match_3_outcome == "H":
                                team_1_points += 3
                            elif match_3_outcome == "D":
                                team_1_points += 1
                                team_3_points += 1
                            else:
                                team_3_points += 3
                            prob *= calculate_outcome_probability(team_1, team_3, match_3_outcome)

                            # Match 4
                            if match_4_outcome == "H":
                                team_2_points += 3
                            elif match_4_outcome == "D":
                                team_2_points += 1
                                team_4_points += 1
                            else:
                                team_4_points += 3
                            prob *= calculate_outcome_probability(team_2, team_4, match_4_outcome)

                            # Match 5
                            if match_5_outcome == "H":
                                team_1_points += 3
                            elif match_5_outcome == "D":
                                team_1_points += 1
                                team_4_points += 1
                            else:
                                team_4_points += 3
                            prob *= calculate_outcome_probability(team_1, team_4, match_5_outcome)

                            # Match 6
                            if match_6_outcome == "H":
                                team_2_points += 3
                            elif match_6_outcome == "D":
                                team_2_points += 1
                                team_3_points += 1
                            else:
                                team_3_points += 3
                            prob *= calculate_outcome_probability(team_2, team_3, match_6_outcome)


                            ## Store results ##
                            if (team_1_points, team_2_points, team_3_points, team_4_points) in outcome_probs:
                                outcome_probs[(team_1_points, team_2_points, team_3_points, team_4_points)] +=  prob
                            else:
                                outcome_probs[(team_1_points, team_2_points, team_3_points, team_4_points)] = prob

    return outcome_probs


def collate_group_probabilities(group_probs, teams):
    """
    Takes output of simulate_group and calculates the distribution for 
    possible final group outcomes.

    Uses the break_ties function to split probabilities when teams
    end on the same number of points.
    """

    # Initialise the collated results
    collated_results = {"Outcomes": [], "Probabilities": []}

    for outcome, p in group_probs.items():

        # Group the teams by their number of points
        point_groups = defaultdict(list)

        current_position = 1

        table = [{}]
        probs = [p]

        for team, pts in zip(teams, outcome):
            point_groups[pts].append(team)

        for pts in sorted(point_groups.keys(), reverse=True):

            # Get the teams with this number of points
            tied_teams = point_groups[pts]

            if len(tied_teams) == 1:
                for tab in table:
                    tab[current_position] = {"team": tied_teams[0], "points": pts}
                current_position += 1
            else:
                permutations = break_ties(tied_teams)

                sub_tables = []
                sub_probs = []

                for ordering, tie_prob in permutations.items():
                    sub_tab = {}
                    for offset, team in enumerate(ordering):
                        sub_tab[current_position+offset] = {"team": team, "points": pts}
                    sub_probs.append(tie_prob)
                    sub_tables.append(sub_tab)


                table_new = []
                probs_new = []

                for tab, prob in zip(table, probs):
                    for sub_tab, sub_prob in zip(sub_tables, sub_probs):
                        table_new.append(tab | sub_tab)
                        probs_new.append(prob*sub_prob)
                table = table_new
                probs = probs_new


                current_position += len(tied_teams)

        for tab, prob in zip(table, probs):
            collated_results["Outcomes"].append(tab)
            collated_results["Probabilities"].append(prob)

    return collated_results
