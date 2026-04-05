
# Import packages
import pandas as pd 
import json
from collections import defaultdict
import numpy as np


## Load data ##

# Elo data
with open('../team_info/initial_elo_ratings.json', 'r') as f:
    elo_ratings = json.load(f)



def break_ties(teams):
    """ 
    Break ties randomly equally for now, can later substitute for a 
    better model.
    """

    import math
    import itertools

    num_sequences = math.factorial(len(teams))

    return {tuple(t_seq): 1/num_sequences for t_seq in itertools.permutations(teams)}


def fill_knockout_slots(group_tables, inverse_groups, third_place_matching):
    """ 
    Takes the results from the group stage and determines which teams
    qualify and where in the knockout bracket they go.
    
    Can have third place teams on same points, so use break_ties to split
    these and determine the respective probabilities. These are then
    sampled from the resulting distribution.
    """

    # import datetime
    # print("Starting to fill knockout slots")
    # start_time = datetime.datetime.now()

    ## Third place teams ##

    third_place_point_groups = defaultdict(list)

    # Get all the third place teams
    for group_result in group_tables.values():
        team, pts = group_result[3]["team"], group_result[3]["points"]
        third_place_point_groups[pts].append(team)

    third_place_teams = []
    # tied_teams_num = 0
    while len(third_place_teams) < 8:

        for pts in sorted(third_place_point_groups.keys(),reverse=True):

            if len(third_place_teams) == 8:
                break
            
            tied_teams = third_place_point_groups[pts]

            if len(tied_teams) + len(third_place_teams) <= 8:
                third_place_teams += tied_teams
            else:
                # tied_teams_num = max(tied_teams_num, len(tied_teams))
                permutations = break_ties(tied_teams)
                perm, prob = zip(*permutations.items())
                
                # Pick from the distribution
                idx = np.random.choice(len(perm),p=prob)
                selected_perm = perm[idx]

                for team in selected_perm:
                    third_place_teams.append(team)
                    if len(third_place_teams) == 8:
                        break


    # end_time_1 = datetime.datetime.now()
    # print(f"To construct third place teams: {end_time_1-start_time}")

    ## Construct the full bracket
                    
    first_second_qualifiers = {
        "1A": group_tables["A"][1]["team"],
        "2A": group_tables["A"][2]["team"],
        "1B": group_tables["B"][1]["team"],
        "2B": group_tables["B"][2]["team"],
        "1C": group_tables["C"][1]["team"],
        "2C": group_tables["C"][2]["team"],
        "1D": group_tables["D"][1]["team"],
        "2D": group_tables["D"][2]["team"],
        "1E": group_tables["E"][1]["team"],
        "2E": group_tables["E"][2]["team"],
        "1F": group_tables["F"][1]["team"],
        "2F": group_tables["F"][2]["team"],
        "1G": group_tables["G"][1]["team"],
        "2G": group_tables["G"][2]["team"],
        "1H": group_tables["H"][1]["team"],
        "2H": group_tables["H"][2]["team"],
        "1I": group_tables["I"][1]["team"],
        "2I": group_tables["I"][2]["team"],
        "1J": group_tables["J"][1]["team"],
        "2J": group_tables["J"][2]["team"],
        "1K": group_tables["K"][1]["team"],
        "2K": group_tables["K"][2]["team"],
        "1L": group_tables["L"][1]["team"],
        "2L": group_tables["L"][2]["team"],
    }
    third_place_qualifiers = arrange_third_place_qualifiers(third_place_teams, inverse_groups, third_place_matching)
    qualified_teams_arrangement = first_second_qualifiers | third_place_qualifiers

    # end_time = datetime.datetime.now()
    # if tied_teams_num >= 8:
    #     print()
    #     print(tied_teams_num)
    #     print(f"To construct third place teams: {end_time_1-start_time}")
    #     print(f"To arrange the teams: {end_time-end_time_1}")
    #     print(f"Total time: {end_time-start_time}")
    #     print("Finished filling knockout slots")
    #     print()
    return qualified_teams_arrangement


    
def arrange_third_place_qualifiers(teams, inv_groups, third_place_matching):
    """
    Arranges the teams that qualified in third place based upon the matching.
    """

    # Determine which groups the teams are from
    qualified_groups = []
    group_match = {}
    for t in teams:
        qualified_groups.append(inv_groups[t])
        group_match[inv_groups[t]] = t
    qualified_groups_str = ''.join(sorted(qualified_groups))

    # Determine who plays who
    third_place_mapping = third_place_matching[third_place_matching["Third-placed teams advanced"]==qualified_groups_str]

    teams_to_play = ["1A","1B","1D","1E","1G","1I","1K","1L"]
    team_matching = {}
    for team in teams_to_play:

        # Find out who they are playing
        opponent = third_place_mapping[team].item()[-1]
        team_matching[f"3_{team}"] = group_match[opponent]
    return team_matching


def calculate_knockout_probabilities(team_a, team_b):
    """
    Returns the probability of the two teams progressing through
    this knockout round.
    """

    # Get the elo scores
    elo_a = elo_ratings[team_a]
    elo_b = elo_ratings[team_b]

    # Calculate the scores 
    Q_a = 10 ** (elo_a/400)
    Q_b = 10 ** (elo_b/400)

    # Calculate the win probabilities
    p_a = Q_a / (Q_a + Q_b)
    p_b = 1 - p_a
    
    return [p_a, p_b]


def simulate_knockout_rounds(knockout_slots, knockout_matches, simulation_results):
    """ 
    Simulates the knockout rounds.
    
    Uses calculate_win_probability 
    """

    # Simulate the matches
    for match_name, match_teams in knockout_matches.items():
        
        team_1, team_2 = knockout_slots[match_teams[0]], knockout_slots[match_teams[1]]
        
        # Calculate the win probabilities
        p1, p2 = calculate_knockout_probabilities(team_1, team_2)

        # Pick a winner
        idx = np.random.choice(2,p=[p1,p2])
        winner = [team_1, team_2][idx]

        knockout_slots[match_name] = winner

        if match_name in ["SF_1","SF_2"]:
            knockout_slots[f"{match_name}_loser"] = [team_1, team_2][(idx+1)%2]

        # Record the results 
        if "R32" in match_name:
            simulation_results[team_1]["R32"] += 1
            simulation_results[team_2]["R32"] += 1
        elif "R16" in match_name:
            simulation_results[team_1]["R16"] += 1
            simulation_results[team_2]["R16"] += 1
        elif "QF" in match_name:
            simulation_results[team_1]["QF"] += 1
            simulation_results[team_2]["QF"] += 1
        elif "SF" in match_name:
            simulation_results[team_1]["SF"] += 1
            simulation_results[team_2]["SF"] += 1
        elif "final" == match_name:
            simulation_results[team_1]["F"] += 1
            simulation_results[team_2]["F"] += 1
            simulation_results[winner]["Winner"] += 1

    return simulation_results


              

