"""' 
Perform the simulation for a (hybrid) Monte Carlo simulation.

The simulation is hybrid since the group stages are still handled
deterministically, and then sampled from their distribution.
"""

import json 
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib as path

from group_stage_engine import simulate_group, collate_group_probabilities
from knockout_stage_engine import fill_knockout_slots, simulate_knockout_rounds


## Define some variables ##

# Run saving
save_dir_name = "test_run_new"
path.Path(f"runs/{save_dir_name}").mkdir(exist_ok=True,parents=True)

# Save frequency
checkpoint_freq = 2000



## Load data ##

# Group data
with open('../team_info/groups.json','r') as f:
    groups = json.load(f)

# Create an inverse group file
inverse_groups = {}
for group_name, teams in groups.items():
    for t in teams:
        inverse_groups[t] = group_name
with open('../team_info/inverse_groups.json',"w") as f:
    json.dump(inverse_groups, f, indent=4)

# Third place matchings
third_place_matching = pd.read_csv('../misc/third_place_table.csv')

# Knockout matches
with open('../team_info/knockout_bracket_template.json','r') as f:
    knockout_matches = json.load(f)

## Group stage ##
    
"""
The group stage is handled more deterministically, computing the probability
distribution across all possible outcomes (order and points). Later, these
outcomes can be sampled from this distribution for a Monte Carlo simulation.
"""

group_stage_results = {} 

for group_name, group_teams in groups.items():
    # print(f"Simulating Group {group_name}")

    group_outcome_probabilities = simulate_group(group_teams)
    group_outcomes = collate_group_probabilities(group_outcome_probabilities, group_teams)
    group_stage_results[group_name] = group_outcomes






## Knockout stage ##
    
""" 
For the knockout stage, perform a Monte Carlo simulation.

Firstly, draw randomly from the distribution of possible group stage outcomes.
Then, follow through the kncokout matches and record all the different outcomes.
"""

num_iterations = 10000
simulation_results = {t: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "Winner": 0} for t in inverse_groups.keys()}

for i in tqdm(range(num_iterations)):


    # Determine the results from the group stage 

    selected_group_stage_outcomes = {}
    for group_name, outcome_dist in group_stage_results.items():
        idx = np.random.choice(len(outcome_dist["Outcomes"]),p=outcome_dist["Probabilities"])
        state = outcome_dist["Outcomes"][idx]
        selected_group_stage_outcomes[group_name] = state



    # Determine which teams go in which knockout slot
    
    knockout_spots = fill_knockout_slots(selected_group_stage_outcomes, inverse_groups, third_place_matching)



    # Simulate the knockout rounds

    simulation_results = simulate_knockout_rounds(knockout_spots, knockout_matches, simulation_results)


    if (i+1)%checkpoint_freq == 0:

        # Format the results for saving
        results_for_saving = {}
        for k, v in simulation_results.items():
            results_for_saving[k] = {v1: v2/(i+1) for v1, v2 in v.items()}

        # Save the results
        with open(f"runs/{save_dir_name}/checkpoint_{i+1}.json","w") as f:
            json.dump(results_for_saving, f, indent=4)


for k, v in simulation_results.items():
    print(f"{k}: {v}")

# Save the final results
results_for_saving = {}
for k, v in simulation_results.items():
    results_for_saving[k] = {v1: v2/(i+1) for v1, v2 in v.items()}

with open(f"runs/{save_dir_name}/final_results.json","w") as f:
    json.dump(results_for_saving, f, indent=4)

