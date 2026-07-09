# Import packages
import json 
import pandas as pd
import numpy as np
from itertools import combinations


with open('../data/inputs/group_stage_schedule.json','r') as f:
    group_stage_schedule = json.load(f)

with open('../data/inputs/fifa_rankings.json','r') as f:
    fifa_rankings = json.load(f)

third_place_matching = pd.read_csv('../data/reference/third_place_table.csv')

rng = np.random.default_rng(seed=42)

class Counter():
    def __init__(self):
        self.count = 0
    
    def increase(self, amt=1):
        self.count += amt
    
    def reset(self):
        self.count = 0

conduct_skip_counter = Counter()

def simulate_group_stage(model, group_tables, team_stats, inverse_groups):
    """
    Simulates the entire group stage.
    - Model: the XGBoost regressor used to predict scorelines
    - group_tables: the tables in the group stages
    - team_stats: the initial values of Elo and other EMA statistics
    """

    EMA_ALPHA_SHORT = team_stats['alpha_values']['short']
    EMA_ALPHA_LONG = team_stats['alpha_values']['long']

    match_results = {}  # May be needed to break ties

    # Go through the matches in order
    for i in range(1,73):

        # Get the match info
        match_info = group_stage_schedule[str(i)]
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        home_adv = match_info['home_adv']
        group_name = inverse_groups[home_team]

        # Get the team statistics
        home_elo = team_stats[home_team]['Elo']
        home_goals_scored_ema_short = team_stats[home_team]['Goals Scored EMA Short']
        home_goals_scored_ema_long = team_stats[home_team]['Goals Scored EMA Long']
        home_goals_conceeded_ema_short = team_stats[home_team]['Goals Conceeded EMA Short']
        home_goals_conceeded_ema_long = team_stats[home_team]['Goals Conceeded EMA Long']

        away_elo = team_stats[away_team]['Elo']
        away_goals_scored_ema_short = team_stats[away_team]['Goals Scored EMA Short']
        away_goals_scored_ema_long = team_stats[away_team]['Goals Scored EMA Long']
        away_goals_conceeded_ema_short = team_stats[away_team]['Goals Conceeded EMA Short']
        away_goals_conceeded_ema_long = team_stats[away_team]['Goals Conceeded EMA Long']

        # Make a pandas dataframe
        match_info_df = pd.DataFrame({
            'home_elo': [home_elo],
            'away_elo': [away_elo],
            'home_goals_scored_ema_short': [home_goals_scored_ema_short],
            'away_goals_scored_ema_short': [away_goals_scored_ema_short],
            'home_goals_conceeded_ema_short': [home_goals_conceeded_ema_short],
            'away_goals_conceeded_ema_short': [away_goals_conceeded_ema_short],
            'home_goals_scored_ema_long': [home_goals_scored_ema_long],
            'away_goals_scored_ema_long': [away_goals_scored_ema_long],
            'home_goals_conceeded_ema_long': [home_goals_conceeded_ema_long],
            'away_goals_conceeded_ema_long': [away_goals_conceeded_ema_long],
            'home_adv': [home_adv]
        })
        match_info_df = mirror_data(match_info_df)
        match_info_df = match_info_df[model.feature_names_in_]  # match feature names and order
        
        # Predict the two teams' expected goals
        home_expected_goals, away_expected_goals = model.predict(match_info_df)

        # Randomly sample a score
        home_score = rng.poisson(home_expected_goals)
        away_score = rng.poisson(away_expected_goals)

        match_results[f"{home_team} vs {away_team}"] = {home_team: home_score, away_team: away_score}

        # Update the league table
        if home_score > away_score:
            group_tables[group_name][home_team]['pts'] += 3
            group_tables[group_name][home_team]['wins'] += 1
            home_result, away_result = 1, 0
        elif home_score < away_score:
            group_tables[group_name][away_team]['pts'] += 3
            group_tables[group_name][away_team]['wins'] += 1
            home_result, away_result = 0, 1
        else:
            group_tables[group_name][home_team]['pts'] += 1
            group_tables[group_name][away_team]['pts'] += 1
            home_result, away_result = 0.5, 0.5
        group_tables[group_name][home_team]['gf'] += home_score
        group_tables[group_name][home_team]['ga'] += away_score
        group_tables[group_name][away_team]['gf'] += away_score
        group_tables[group_name][away_team]['ga'] += home_score


        ## Update team statistics ##

        # Elo scores
        if home_adv == 1:
            scaled_home_elo = home_elo + 1
            scaled_away_elo = away_elo
        elif home_adv == -1:
            scaled_home_elo = home_elo
            scaled_away_elo = away_elo + 1
        else:
            scaled_home_elo = home_elo 
            scaled_away_elo = away_elo

        # Expected results
        er_home = 1 / (10**(-(scaled_home_elo-scaled_away_elo)/400)+1)
        er_away = 1 - er_home

        # Modify weighting
        weight = 60 # Default for World Cup finals
        goal_diff = abs(home_score - away_score)
        if goal_diff >= 4:
            weight *= (1 + 3/4 + (goal_diff - 3)/8)
        elif goal_diff == 3:
            weight *= 1 + 3/4
        elif goal_diff == 2:
            weight *= 1 + 1/2

        # Calculate new Elo values
        new_home_elo = home_elo + weight * (home_result - er_home)
        new_away_elo = away_elo + weight * (away_result - er_away)

        # Goals EMAs
        new_home_scored_ema_short = EMA_ALPHA_SHORT * home_score + (1-EMA_ALPHA_SHORT) * home_goals_scored_ema_short
        new_home_conceeded_ema_short = EMA_ALPHA_SHORT * away_score + (1-EMA_ALPHA_SHORT) * home_goals_conceeded_ema_short
        new_away_scored_ema_short = EMA_ALPHA_SHORT * away_score + (1-EMA_ALPHA_SHORT) * away_goals_scored_ema_short 
        new_away_conceeded_ema_short = EMA_ALPHA_SHORT * home_score + (1-EMA_ALPHA_SHORT) * away_goals_conceeded_ema_short 

        new_home_scored_ema_long = EMA_ALPHA_LONG * home_score + (1-EMA_ALPHA_LONG) * home_goals_scored_ema_long
        new_home_conceeded_ema_long = EMA_ALPHA_LONG * away_score + (1-EMA_ALPHA_LONG) * home_goals_conceeded_ema_long 
        new_away_scored_ema_long = EMA_ALPHA_LONG * away_score + (1-EMA_ALPHA_LONG) * away_goals_scored_ema_long
        new_away_conceeded_ema_long = EMA_ALPHA_LONG * home_score + (1-EMA_ALPHA_LONG) * away_goals_conceeded_ema_long

        # Update the team statistics
        team_stats[home_team] = {
            "Elo": new_home_elo,
            "Num. Matches": team_stats[home_team]['Num. Matches'] + 1,
            "Goals Scored EMA Short": new_home_scored_ema_short,
            "Goals Conceeded EMA Short": new_home_conceeded_ema_short,
            "Goals Scored EMA Long": new_home_scored_ema_long,
            "Goals Conceeded EMA Long": new_home_conceeded_ema_long
        }
        team_stats[away_team] = {
            "Elo": new_away_elo,
            "Num. Matches": team_stats[away_team]['Num. Matches'] + 1,
            "Goals Scored EMA Short": new_away_scored_ema_short,
            "Goals Conceeded EMA Short": new_away_conceeded_ema_short,
            "Goals Scored EMA Long": new_away_scored_ema_long,
            "Goals Conceeded EMA Long": new_away_conceeded_ema_long
        }


    return group_tables, team_stats, match_results


def fill_knockout_slots(group_tables, match_results, inverse_groups):
    """
    Fill the knockout slots based upon the final state of the group
    tables.
    """
    
    final_group_standings = {}

    # Determine the standings within each group
    for group_name, table in group_tables.items():
        group_order = determine_group_order(table, match_results)
        final_group_standings[group_name] = group_order
    
    # Begin filling knockout slots
    first_second_qualifiers = {
        "1A": final_group_standings["A"][0],
        "2A": final_group_standings["A"][1],
        "1B": final_group_standings["B"][0],
        "2B": final_group_standings["B"][1],
        "1C": final_group_standings["C"][0],
        "2C": final_group_standings["C"][1],
        "1D": final_group_standings["D"][0],
        "2D": final_group_standings["D"][1],
        "1E": final_group_standings["E"][0],
        "2E": final_group_standings["E"][1],
        "1F": final_group_standings["F"][0],
        "2F": final_group_standings["F"][1],
        "1G": final_group_standings["G"][0],
        "2G": final_group_standings["G"][1],
        "1H": final_group_standings["H"][0],
        "2H": final_group_standings["H"][1],
        "1I": final_group_standings["I"][0],
        "2I": final_group_standings["I"][1],
        "1J": final_group_standings["J"][0],
        "2J": final_group_standings["J"][1],
        "1K": final_group_standings["K"][0],
        "2K": final_group_standings["K"][1],
        "1L": final_group_standings["L"][0],
        "2L": final_group_standings["L"][1],
    }

    # Determine the third place qualifiers
    third_place_qualifiers = determine_third_place_qualifiers(group_tables, final_group_standings, inverse_groups)
    qualified_teams_arrangement = first_second_qualifiers | third_place_qualifiers

    return qualified_teams_arrangement


def determine_group_order(group_table, match_results):
    """
    Orders the teams within a group.
    """
    teams_ordered = []

    # Extract the points scored by each team
    pts = {}
    for team_name, team_res in group_table.items():
        pts[team_res['pts']] = pts.get(team_res['pts'],[]) + [team_name]
    pts = {k: pts[k] for k in sorted(pts.keys(), reverse=True)}
    
    # Check for tied teams
    for v in pts.values():
        if len(v) > 1:
            ordered_subset = break_tied_teams(v, group_table, match_results)
            for t in ordered_subset:
                teams_ordered.append(t)
        else:
            # No tie
            teams_ordered.append(v[0])

    return teams_ordered


def break_tied_teams(tied_teams, group_table, match_results):
    """ 
    Break ties of teams on same number of points in group.
    """

    num_teams = len(tied_teams)
    teams_ordered = []

    ##### Step 1 #####
    """
    Mini table of tied teams, ordered by points, GD, GF in order
    """

    # Create mini table
    mini_table = {t: {'pts': 0, 'gf': 0, 'ga': 0} for t in tied_teams}
    for teams in combinations(tied_teams, 2):
        team_1, team_2 = teams

        # Get the result of their match
        if f"{team_1} vs {team_2}" in match_results:
            match_score = match_results[f"{team_1} vs {team_2}"]
        else:
            match_score = match_results[f"{team_2} vs {team_1}"]

        team_1_goals = match_score[team_1]
        team_2_goals = match_score[team_2]

        # Update the table
        if team_1_goals > team_2_goals:
            mini_table[team_1]['pts'] += 3
        elif team_1_goals < team_2_goals:
            mini_table[team_2]['pts'] += 3
        else:
            mini_table[team_1]['pts'] += 1
            mini_table[team_2]['pts'] += 1
        mini_table[team_1]['gf'] += team_1_goals
        mini_table[team_1]['ga'] += team_2_goals
        mini_table[team_2]['gf'] += team_2_goals
        mini_table[team_2]['ga'] += team_1_goals

    # Extract the points scored by each team
    pts = {}
    for team_name, team_res in mini_table.items():
        pts[team_res['pts']] = pts.get(team_res['pts'],[]) + [team_name]
    pts = {k: pts[k] for k in sorted(pts.keys(), reverse=True)}

    # Order the teams
    for points_scored, teams in pts.items():
        if len(teams) > 1:
            # Have a tie based on points, now move to GD

            # Calculate goal differences
            gd = {}
            for t in teams:
                goal_diff = mini_table[t]['gf'] - mini_table[t]['ga']
                gd[t] = gd.get(goal_diff,[]) + [t]
            gd = {k: gd[k] for k in sorted(gd.keys(), reverse=True)}

            # Order by goal difference
            for goal_diff, teams_by_gd in gd.items():
                if len(teams_by_gd) > 1:
                    # Still tied, now GF

                    # Calculate goals scored
                    gf = {}
                    for t in teams_by_gd:
                        gf[t] = gf.get(mini_table[t]['gf'],[]) + [mini_table[t]['gf']]
                    gf = {k: gd[k] for k in sorted(gd.keys(), reverse=True)}

                    # Order by GF
                    for goals_scored, teams_by_gf in gf.items():
                        if len(teams_by_gf) > 1:
                            # Still tied by GF
                            if len(teams_by_gf) < num_teams:
                                # Repeat the same procedure
                                ord_subset = break_tied_teams(teams_by_gf, group_table, match_results)
                                for t in ord_subset:
                                    teams_ordered.append(t)
                            else:
                                # Failed to separate teams, move to step 2
                                ord_subset = break_tied_teams_step_2(teams_by_gf, group_table, match_results)
                                for t in ord_subset:
                                    teams_ordered.append(t)

                        else:
                            teams_ordered.append(teams_by_gf[0])
                
                else:
                    teams_ordered.append(teams_by_gd[0])

        else:
            # No longer tied
            teams_ordered.append(teams[0])

    return teams_ordered


def break_tied_teams_step_2(tied_teams, group_table, match_results):
    """
    Break ties that survived step 1 of breaking
    """

    teams_ordered = []

    ##### Step 1 #####
    """
    Order by GD, GF in order in the full group table.
    Does then order by conduct score. This is not simulated,
    and so is skipped, but any instance is counted.
    """

    # Calculate goal differences
    gd = {}
    for t in tied_teams:
        goal_diff = group_table[t]['gf'] - group_table[t]['ga']
        gd[t] = gd.get(goal_diff,[]) + [t]
    gd = {k: gd[k] for k in sorted(gd.keys(), reverse=True)}

    # Order by GD
    for teams_by_gd in gd.values():
        if len(teams_by_gd) > 1:
            # Still tied, now GF

            # Calculate goals scored
            gf = {}
            for t in teams_by_gd:
                gf[t] = gf.get(group_table[t]['gf'],[]) + [group_table[t]['gf']]
            gf = {k: gd[k] for k in sorted(gd.keys(), reverse=True)}

            # Order by GF
            for teams_by_gf in gf.values():
                if len(teams_by_gf) > 1:
                    # Still tied, skip conduct score and go to FIFA rankings
                    conduct_skip_counter.increase()

                    # Get the rankings for each of the teams
                    rankings = []
                    for t in teams_by_gf:
                        rankings.append(fifa_rankings[t])
                    ord_subset = [t for _, t in sorted(zip(rankings, teams_by_gf))]
                    for t in ord_subset:
                        teams_ordered.append(t)
                
                else:
                    teams_ordered.append(teams_by_gf[0])

        else:
            teams_ordered.append(teams_by_gd[0])

    return teams_ordered


def determine_third_place_qualifiers(group_tables, final_group_standings, inverse_groups):
    """
    Determine which third place teams qualify given the group tables
    """

    teams_ordered = []

    # Construct a league table of third place teams
    third_place_league_table = {}
    for group_name, group_standings in final_group_standings.items():
        
        third_team = group_standings[2]

        # Add to league table
        third_place_league_table[third_team] = group_tables[group_name][third_team]

    # Start by ordering by points scored
    pts = {}
    for team_name, team_results in third_place_league_table.items():
        pts[team_results['pts']] = pts.get(team_results['pts'],[]) + [team_name]
    pts = {k: pts[k] for k in sorted(pts.keys(), reverse=True)}

    for teams_by_pts in pts.values():
        if len(teams_by_pts) > 1:
            # Ties, break by GD

            # Calculate goal difference
            gd = {}
            for t in teams_by_pts:
                goal_diff = third_place_league_table[t]['gf'] - third_place_league_table[t]['ga']
                gd[goal_diff] = gd.get(goal_diff, []) + [t]
            gd = {k: gd[k] for k in sorted(gd.keys(), reverse=True)}

            # Order by goal difference
            for teams_by_gd in gd.values():
                if len(teams_by_gd) > 1:
                    # Tied, break by GF
                    
                    # Calculate GF
                    gf = {}
                    for t in teams_by_gd:
                        gf[third_place_league_table[t]['gf']] = gf.get(third_place_league_table[t]['gf'],[]) + [t]
                    gf = {k: gf[k] for k in sorted(gf.keys(), reverse=True)}

                    # Order by GF
                    for teams_by_gf in gf.values():
                        if len(teams_by_gf) > 1:
                            # Tied, skip conduct score, use FIFA rankings
                            if len(teams_ordered) >=8 or (len(teams_ordered)+len(teams_by_gf))<=8:
                                pass
                            else:
                                # Affects outcome
                                conduct_skip_counter.increase()

                            # Get the rankings for each of the teams
                            rankings = []
                            for t in teams_by_gf:
                                rankings.append(fifa_rankings[t])
                            ord_subset = [t for _, t in sorted(zip(rankings, teams_by_gf))]
                            for t in ord_subset:
                                teams_ordered.append(t)

                        else:
                            teams_ordered.append(teams_by_gf[0])

                else:
                    teams_ordered.append(teams_by_gd[0])

        else:
            teams_ordered.append(teams_by_pts[0])


    ### Arrange the top 8 third place teams into knockout positions ###

    # Extract the qualifying teams
    qualifying_third_place_teams = teams_ordered[:8]

    # Extract their groups
    qualified_groups = [inverse_groups[t] for t in qualifying_third_place_teams]
    qualified_groups = ''.join(sorted(qualified_groups))

    third_place_mapping = third_place_matching[third_place_matching["Third-placed teams advanced"]==qualified_groups]

    teams_to_play = ["1A","1B","1D","1E","1G","1I","1K","1L"]
    team_matching = {}
    for team in teams_to_play:

        # Find out wh they are playing
        opponent = third_place_mapping[team].item()[-1]
        team_matching[f"3_{team}"] = final_group_standings[opponent][2]
    
    return team_matching


def simulate_knockout_rounds(model, team_stats, knockout_slots, knockout_matches, simulation_results):
    """
    Simulate the knockout rounds
    """

    EMA_ALPHA_SHORT = team_stats['alpha_values']['short']
    EMA_ALPHA_LONG = team_stats['alpha_values']['long']

    # Simulate the matches
    for match_name, match_teams in knockout_matches.items():

        home_team, away_team = knockout_slots[match_teams[0]], knockout_slots[match_teams[1]]

        # Get the team statistics
        home_elo = team_stats[home_team]['Elo']
        home_goals_scored_ema_short = team_stats[home_team]['Goals Scored EMA Short']
        home_goals_scored_ema_long = team_stats[home_team]['Goals Scored EMA Long']
        home_goals_conceeded_ema_short = team_stats[home_team]['Goals Conceeded EMA Short']
        home_goals_conceeded_ema_long = team_stats[home_team]['Goals Conceeded EMA Long']

        away_elo = team_stats[away_team]['Elo']
        away_goals_scored_ema_short = team_stats[away_team]['Goals Scored EMA Short']
        away_goals_scored_ema_long = team_stats[away_team]['Goals Scored EMA Long']
        away_goals_conceeded_ema_short = team_stats[away_team]['Goals Conceeded EMA Short']
        away_goals_conceeded_ema_long = team_stats[away_team]['Goals Conceeded EMA Long']

        home_adv = 0    # For now, don't give either team a home advantage

        # Make a pandas dataframe
        match_info_df = pd.DataFrame({
            'home_elo': [home_elo],
            'away_elo': [away_elo],
            'home_goals_scored_ema_short': [home_goals_scored_ema_short],
            'away_goals_scored_ema_short': [away_goals_scored_ema_short],
            'home_goals_conceeded_ema_short': [home_goals_conceeded_ema_short],
            'away_goals_conceeded_ema_short': [away_goals_conceeded_ema_short],
            'home_goals_scored_ema_long': [home_goals_scored_ema_long],
            'away_goals_scored_ema_long': [away_goals_scored_ema_long],
            'home_goals_conceeded_ema_long': [home_goals_conceeded_ema_long],
            'away_goals_conceeded_ema_long': [away_goals_conceeded_ema_long],
            'home_adv': [home_adv]
        })
        match_info_df = mirror_data(match_info_df)
        match_info_df = match_info_df[model.feature_names_in_]  # match feature names and order

        # Predict the two teams' expected goals
        home_expected_goals, away_expected_goals = model.predict(match_info_df)

        # Randomly sample a score
        home_score = rng.poisson(home_expected_goals)
        away_score = rng.poisson(away_expected_goals)

        # Determine the winner
        if home_score > away_score:
            winner = home_team
            loser = away_team
            home_result, away_result = 1, 0
        elif home_score < away_score:
            winner = away_team
            loser = home_team
            home_result, away_result = 0, 1
        else:
            # Extra time
            home_result, away_result = 0.5, 0.5 # Use result from normal time

            # Simulate goals again
            home_score_et = rng.poisson(home_expected_goals/3)
            away_score_et = rng.poisson(away_expected_goals/3)

            if home_score_et > away_score_et:
                winner = home_team
                loser = away_team
            elif home_score_et < away_score_et:
                winner = away_team
                loser = home_team
            else:
                # Level at extra time, randomly sample winner weighted by xG

                # Normalize 
                total_xg = home_expected_goals + away_expected_goals
                probs = [home_expected_goals / total_xg, away_expected_goals / total_xg]

                probs = np.array([home_expected_goals, away_expected_goals], dtype=np.float64)
                probs /= probs.sum()

                # Draw the winner
                winner = rng.choice([home_team, away_team], p=probs)
                loser = [home_team, away_team].pop([home_team,away_team].index(winner))


        # Fill the bracket
        knockout_slots[match_name] = winner

        if match_name in ["SF_1","SF_2"]:
            knockout_slots[f"{match_name}_loser"] = loser


        ## Update team statistics ##
            
        # Elo scores
        if home_adv == 1:
            scaled_home_elo = home_elo + 1
            scaled_away_elo = away_elo
        elif home_adv == -1:
            scaled_home_elo = home_elo
            scaled_away_elo = away_elo + 1
        else:
            scaled_home_elo = home_elo 
            scaled_away_elo = away_elo

        # Expected results
        er_home = 1 / (10**(-(scaled_home_elo-scaled_away_elo)/400)+1)
        er_away = 1 - er_home

        # Modify weighting
        weight = 60 # Default for World Cup finals
        goal_diff = abs(home_score - away_score)
        if goal_diff >= 4:
            weight *= (1 + 3/4 + (goal_diff - 3)/8)
        elif goal_diff == 3:
            weight *= 1 + 3/4
        elif goal_diff == 2:
            weight *= 1 + 1/2

        # Calculate new Elo values
        new_home_elo = home_elo + weight * (home_result - er_home)
        new_away_elo = away_elo + weight * (away_result - er_away)

        # Goals EMAs
        new_home_scored_ema_short = EMA_ALPHA_SHORT * home_score + (1-EMA_ALPHA_SHORT) * home_goals_scored_ema_short
        new_home_conceeded_ema_short = EMA_ALPHA_SHORT * away_score + (1-EMA_ALPHA_SHORT) * home_goals_conceeded_ema_short
        new_away_scored_ema_short = EMA_ALPHA_SHORT * away_score + (1-EMA_ALPHA_SHORT) * away_goals_scored_ema_short 
        new_away_conceeded_ema_short = EMA_ALPHA_SHORT * home_score + (1-EMA_ALPHA_SHORT) * away_goals_conceeded_ema_short 

        new_home_scored_ema_long = EMA_ALPHA_LONG * home_score + (1-EMA_ALPHA_LONG) * home_goals_scored_ema_long
        new_home_conceeded_ema_long = EMA_ALPHA_LONG * away_score + (1-EMA_ALPHA_LONG) * home_goals_conceeded_ema_long 
        new_away_scored_ema_long = EMA_ALPHA_LONG * away_score + (1-EMA_ALPHA_LONG) * away_goals_scored_ema_long
        new_away_conceeded_ema_long = EMA_ALPHA_LONG * home_score + (1-EMA_ALPHA_LONG) * away_goals_conceeded_ema_long

        # Update the team statistics
        team_stats[home_team] = {
            "Elo": new_home_elo,
            "Num. Matches": team_stats[home_team]['Num. Matches'] + 1,
            "Goals Scored EMA Short": new_home_scored_ema_short,
            "Goals Conceeded EMA Short": new_home_conceeded_ema_short,
            "Goals Scored EMA Long": new_home_scored_ema_long,
            "Goals Conceeded EMA Long": new_home_conceeded_ema_long
        }
        team_stats[away_team] = {
            "Elo": new_away_elo,
            "Num. Matches": team_stats[away_team]['Num. Matches'] + 1,
            "Goals Scored EMA Short": new_away_scored_ema_short,
            "Goals Conceeded EMA Short": new_away_conceeded_ema_short,
            "Goals Scored EMA Long": new_away_scored_ema_long,
            "Goals Conceeded EMA Long": new_away_conceeded_ema_long
        }



        
        # Record the results 
        if "R32" in match_name:
            simulation_results[home_team]["R32"] += 1
            simulation_results[away_team]["R32"] += 1
        elif "R16" in match_name:
            simulation_results[home_team]["R16"] += 1
            simulation_results[away_team]["R16"] += 1
        elif "QF" in match_name:
            simulation_results[home_team]["QF"] += 1
            simulation_results[away_team]["QF"] += 1
        elif "SF" in match_name:
            simulation_results[home_team]["SF"] += 1
            simulation_results[away_team]["SF"] += 1
        elif "final" == match_name:
            simulation_results[home_team]["F"] += 1
            simulation_results[away_team]["F"] += 1
            simulation_results[winner]["Winner"] += 1


    return simulation_results
    
    



def mirror_data(input_df, merge_dataframes=True):

    # Define the pairs to swap
    swap_pairs = [
        ('home_team', 'away_team'),
        ('home_score', 'away_score'),
        ('home_elo', 'away_elo'),
        ('home_elo_num_matches', 'away_elo_num_matches'),
        ('home_goals_scored_ema_short', 'away_goals_scored_ema_short'),
        ('home_goals_conceeded_ema_short', 'away_goals_conceeded_ema_short'),
        ('home_goals_scored_ema_long', 'away_goals_scored_ema_long'),
        ('home_goals_conceeded_ema_long', 'away_goals_conceeded_ema_long'),
    ]

    # Create a rename map
    rename_map = {}
    for home_col, away_col in swap_pairs:
        rename_map[home_col] = away_col
        rename_map[away_col] = home_col

    # Create a copy of the dataframe
    df_mirrored = input_df.copy()

    # Rename columns
    df_mirrored = df_mirrored.rename(columns=rename_map)

    # Flip the winner
    if 'winner' in df_mirrored:
        df_mirrored['winner'] = 2 - df_mirrored['winner']

    # Flip the home advantage
    df_mirrored['home_adv'] = -1 * df_mirrored['home_adv']

    # Combine together
    if merge_dataframes:
        combined_results_df = pd.concat([input_df, df_mirrored], axis=0)
        # combined_results_df = combined_results_df.sort_values('date').reset_index(drop=True)
    else:
        combined_results_df = df_mirrored
    

    # Create a match ID column
    combined_results_df = combined_results_df.rename(columns={"Unnamed: 0": "match_id"})

    return combined_results_df

