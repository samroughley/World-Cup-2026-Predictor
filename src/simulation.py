"""
Perform Monte Carlo simulation.

This simulation is a full Monte Carlo simulation, where the full tournament
is simulated many times, rather than the group stage being handled
deterministically.
"""

# Import packages
import argparse
import json 
import xgboost as xgb
import pandas as pd
import copy 
from tqdm import tqdm
import pathlib as path
import re


from engine import simulate_group_stage, fill_knockout_slots, simulate_knockout_rounds, conduct_skip_counter


def parse_arguments():
    """
    Handles command line arguments
    """

    parser = argparse.ArgumentParser(description="Run tournament simulations.")

    parser.add_argument(
        '--run_name',
        type=str,
        default='example_run',
        help='Name of the simulation run folder'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Total number of simulation iterations'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=100,
        help='How many iterations before saving a checkpoint'
    )

    return parser.parse_args()

def main():

    # Parse CLI arguments
    args = parse_arguments()

    NUM_ITERATIONS = args.iterations
    CHECKPOINT_FREQ = args.checkpoint_freq
    save_dir_name = args.run_name

    checkpoint_dir = path.Path(f'../data/outputs/{save_dir_name}/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    ## Load Data and Models ##

    # Team statistics
    with open('../data/inputs/latest_statistics.json','r') as f:
        latest_stats = json.load(f)

    # Model
    model = xgb.XGBRegressor()
    model.load_model('models/XGBoost_regressor_model.json')

    # Group data
    with open('../data/inputs/groups.json','r') as f:
        groups = json.load(f)

    # Create an inverse group file
    inverse_groups = {}
    for group_name, teams in groups.items():
        for t in teams:
            inverse_groups[t] = group_name
    # with open('../team_info/inverse_groups.json',"w") as f:
    #     json.dump(inverse_groups, f, indent=4)

    # Third place matchings
    third_place_matching = pd.read_csv('../data/reference/third_place_table.csv')

    # Knockout matches
    with open('../data/inputs/knockout_bracket_template.json','r') as f:
        knockout_matches = json.load(f)


    conducts_skipped = 0
    start_iteration = 0
    simulation_results = {t: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "Winner": 0} for t in inverse_groups.keys()}


    # Check if checkpoint exists
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
    if checkpoint_files:

        # Extract iteration numbers
        checkpoint_iters = []
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)\.json', f.name)
            if match:
                checkpoint_iters.append((int(match.group(1)), f))
        
        if checkpoint_iters:
            # Get the checkpoint with the highest iteration number
            latest_iter, latest_file = max(checkpoint_iters, key=lambda x: x[0])
            
            print(f"--> Found existing checkpoints. Resuming from {latest_file.name}...")

            with open(latest_file, 'r') as f:
                saved_data = json.load(f)
            
            # Extract conducted skips
            conducts_skipped = saved_data.pop('conduct_skip_counter', 0)
            
            # REVERSE THE AVERAGING: Convert saved decimals back to raw simulation counts
            for team, stages in saved_data.items():
                if team in simulation_results:
                    for stage, average_val in stages.items():
                        simulation_results[team][stage] = round(average_val * latest_iter)
            
            # Set loop start point
            start_iteration = latest_iter


    ## Simulation ##
        
    for i in tqdm(range(start_iteration, NUM_ITERATIONS)):

        # Initialise group tables
        group_tables = {group_name: {t: {'pts': 0, 'gf': 0, 'ga': 0, 'wins': 0} for t in group_teams} for group_name, group_teams in groups.items()}

        # Create a copy of the latest stats
        team_stats = copy.deepcopy(latest_stats)    # Can update during the simulation without interfering with other runs

        # Simulate the group stage
        group_tables, team_stats, match_results = simulate_group_stage(model, group_tables, team_stats, inverse_groups)

        # Fill the knockout slots
        knockout_configuration = fill_knockout_slots(group_tables, match_results, inverse_groups)

        # Simulate the knockout rounds
        simulation_results = simulate_knockout_rounds(model, team_stats, knockout_configuration, knockout_matches, simulation_results)

        if conduct_skip_counter.count >= 1:
            conducts_skipped += 1
        conduct_skip_counter.reset()

        if (i+1)%CHECKPOINT_FREQ == 0:

            # Format the results for saving
            results_for_saving = {}
            for k, v in simulation_results.items():
                results_for_saving[k] = {v1: v2/(i+1) for v1, v2 in v.items()}
            results_for_saving['conduct_skip_counter'] = conducts_skipped

            # Save the results
            with open(f"../data/outputs/{save_dir_name}/checkpoints/checkpoint_{i+1}.json","w") as f:
                json.dump(results_for_saving, f, indent=4)


    for k, v in simulation_results.items():
        print(f"{k}: {v}")

    # Save the final results
    results_for_saving = {}
    for k, v in simulation_results.items():
        results_for_saving[k] = {v1: v2/(max(start_iteration,NUM_ITERATIONS)) for v1, v2 in v.items()}
    results_for_saving['conduct_skip_counter'] = conducts_skipped

    with open(f"../data/outputs/{save_dir_name}/final_results.json","w") as f:
        json.dump(results_for_saving, f, indent=4)


    
    ## Save a processed csv file ##

    # Load additional data

    # Opta predictions
    opta_preds_df = pd.read_csv('../data/reference/Opta_Predictions.csv')

    # Team statistics
    with open('../data/inputs/latest_statistics.json','r') as f:
        team_stats = json.load(f)
    
    elo_df = pd.DataFrame({
        'Team': [k for k in team_stats.keys() if k != 'alpha_values'],
        'Elo': [v['Elo'] for v in team_stats.values() if 'Elo' in v]
    })

    results_for_saving.pop('conduct_skip_counter')
    sim_df = pd.DataFrame.from_dict(results_for_saving, orient='index').reset_index()
    sim_df.columns = ['Team','R32_mine','R16_mine','QF_mine','SF_mine','F_mine','Winner_mine']
    # Make %
    cols_to_fix = [c for c in sim_df.columns if '_mine' in c]
    sim_df[cols_to_fix] = sim_df[cols_to_fix] * 100

    opta_preds_df = opta_preds_df.rename(columns={
        'R16': 'R16_Opta', 'QF': 'QF_Opta', 'SF': 'SF_Opta', 
        'Final': 'F_Opta', 'Winner': 'Winner_Opta'
    })

    # Merge info
    final_df = pd.merge(sim_df, opta_preds_df, on='Team', how='left')
    final_df = pd.merge(final_df, elo_df, on='Team', how='left')

    final_df['R32_Opta'] = None # Don't publish this

    final_df.to_csv(f'../data/outputs/{save_dir_name}/processed_results.csv', index=False)


if __name__ == "__main__":
    main()