"""
Get the latest ELO ratings for the different teams.
"""

# Import packages
import requests
import pandas as pd
import io
import json

def get_eloratings_data():
    # 1. Fetch the Team Name Mapping (The "Decoder Ring")
    # This JSON maps IDs like "105" to "Argentina"

    mapping_url = "https://www.eloratings.net/teams.json"
    mapping_url = "https://www.eloratings.net/en.teams.tsv?_=1775301283918"
    headers = {"User-Agent": "Mozilla/5.0"} # Good practice to include
    
    mapping_res = requests.get(mapping_url, headers=headers)

    # team_mapping = mapping_res.json()
    map_df = pd.read_csv(io.StringIO(mapping_res.text), sep='\t', header=None, usecols=[0, 1], on_bad_lines='skip')
    team_dict = dict(zip(map_df[0], map_df[1]))
    
    # 2. Fetch the Ratings TSV
    # World.tsv contains the raw data for all teams
    tsv_url = "https://www.eloratings.net/World.tsv"
    tsv_res = requests.get(tsv_url, headers=headers)
    
    # 3. Load into Pandas
    # This file has no headers and uses Tabs (\t)
    df = pd.read_csv(io.StringIO(tsv_res.text), sep='\t', header=None)

    # 4. Define and Clean the columns
    # Based on the site's data structure:
    df = df.rename(columns={2: "Team ID", 3: "Elo"})

    # 5. Map the TeamID to the actual Name
    # We ensure TeamID is a string to match the JSON keys
    df['Team Name'] = df['Team ID'].astype(str).map(team_dict)

    # 6. Final Polish
    # Filter for the main columns and return
    # return df[['Rank', 'Team Name', 'Elo']]
    # return df[['Team Name',3]]
    return df[['Team Name','Elo']]


# Execute
try:
    elo_df = get_eloratings_data()
    print("--- Top 10 World Rankings ---")
    print(elo_df.head(10).to_string(index=False))

    # Make a json
    elo_json = {}
    for _, row in elo_df.iterrows():
        if row['Team Name'] == 'Curacao':
            print(row['Team Name'])
        elo_json[row['Team Name']] = row['Elo']

    with open("initial_elo_ratings.json","w") as f:
        json.dump(elo_json, f, indent=4)
    
    # Save a local copy
    # elo_df.to_csv("football_elo_ratings.csv", index=False)
    print("\nLocal copy updated: initial_elo_ratings.json")
    
except Exception as e:
    print(f"Update failed: {e}")