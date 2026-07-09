import streamlit as st 
import pandas as pd 
import json 

# Create page
st.set_page_config(layout="wide",
                   page_title="WC 2026 Predictor",
                   page_icon="🏆")


## Load Data ##

with open("config.json","r") as f:
    config_file = json.load(f)

# Define the name of the run (boilerplate)
RUN_NAME = "test_run"

# My Predictions
# with open("Monte Carlo Simulation/runs/quick_run/final_results.json","r") as f:
# with open("Advanced Monte Carlo Simulation/runs/test_run/final_results.json","r") as f:
with open(f"data/outputs/{RUN_NAME}/final_results.json","r") as f:
    simulation_results = json.load(f)
    simulation_results.pop('conduct_skip_counter')

    # Rename
    for team, outcomes in simulation_results.items():
        simulation_results[team]["Final"] = simulation_results[team]["F"]
        del simulation_results[team]["F"]



# Opta's Predictions
opta_preds = pd.read_csv('data/reference/Opta_Predictions.csv')[['Team','Group','R16','QF','SF','Final','Winner']]
opta_preds[['Group','R16','QF','SF','Final','Winner']] = opta_preds[['Group','R16','QF','SF','Final','Winner']].astype('float')
# opta_preds = opta_preds.rename(columns={"R32_Exit": "R32"})


# Prepare Data 
data = {
    "Team": [],
    "R32_mine": [],
    "R32_Opta": [],
    "R16_mine": [],
    "R16_Opta": [],
    "QF_mine": [],
    "QF_Opta": [],
    "SF_mine": [],
    "SF_Opta": [],
    "Final_mine": [],
    "Final_Opta": [],
    "Winner_mine": [],
    "Winner_Opta": [],
}

stages = ["R32","R16","QF","SF","Final","Winner"]
for team, outcomes in simulation_results.items():
    data["Team"].append(team)

    for out, prob in outcomes.items():
        data[f"{out}_mine"].append(prob*100)

        try:
            opta_prediction = opta_preds[opta_preds["Team"]==team][out].values
        except:
            opta_prediction = []

        if len(opta_prediction) > 0:
            data[f"{out}_Opta"].append(opta_prediction[0])
        else:
            data[f"{out}_Opta"].append(-1)



df = pd.DataFrame(data)

# Sort the dataframe
df = df.sort_values('Winner_mine', ascending=False)

# Create the combined display column
def combine_probs(row, my_col, opta_col):
    return f"{row[my_col]:.2f}% <span class='opta-text'>({row[opta_col]:.2f}%)</span>"

# Apply this to your knockout columns
for stage in stages:
    if config_file['opta_comparison']:
        df[stage] = df.apply(combine_probs, axis=1, args=(f"{stage}_mine", f"{stage}_Opta"))
    else:
        df[stage] = df[f"{stage}_mine"].round(decimals=2).astype(str) + '%'

# Custom CSS to make the bracketed text faint
# st.markdown("""
#     <style>
#     .opta-text {
#         color: #808080; /* Grey color */
#         font-size: 0.85em;
#         font-weight: 300;
#     }
#     table {
#         width: 100%;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
st.markdown("""
    <style>
    /* Style the whole table */
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
    }
    th {
        background-color: white;
        color: black;
        text-align: left;
        padding: 12px;
    }
    td {
        padding: 10px;
        border-bottom: 1px solid #444;
    }
    /* Style the Opta numbers */
    .opta-text {
        color: #888888;
        font-size: 0.85em;
        margin-left: 5px;
    }
    /* Bold the Team Name column specifically */
    td:first-child {
        font-weight: bold;
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)





## Create Dashboard ##

with st.sidebar:
    st.markdown(f"*Version Number: {config_file['version_num']}*")


st.title("🏆 2026 World Cup Probability Dashboard")

st.markdown("""
Predicted probabilities of each team reaching each knockout round in the Men's 2026 FIFA World Cup.
""")

# A mapping dictionary
flag_map = {
    "Argentina": "🇦🇷",
    "Brazil": "🇧🇷",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷",
    "USA": "🇺🇸",
    "Mexico": "🇲🇽",
    "Portugal": "🇵🇹",
    "Spain": "🇪🇸",
    "Netherlands": "🇳🇱",
    "Colombia": "🇨🇴",
    "Belgium": "🇧🇪",
    "Croatia": "🇭🇷",
    "Senegal": "🇸🇳",
    "Germany": "🇩🇪",
    "Morocco": "🇲🇦",
    "Japan": "🇯🇵",
    "Turkey": "🇹🇷",
    "Norway": "🇳🇴",
    "Switzerland": "🇨🇭",
    "Uruguay": "🇺🇾",
    "Austria": "🇦🇹",
    "Canada": "🇨🇦",
    "Panama": "🇵🇦",
    "Ecuador": "🇪🇨",
    "Algeria": "🇩🇿",
    "Uzbekistan": "🇺🇿",
    "Australia": "🇦🇺",
    "Paraguay": "🇵🇾",
    "South Korea": "🇰🇷",
    "Sweden": "🇸🇪",
    "Tunisia": "🇹🇳",
    "Ivory Coast": "🇨🇮",
    "United States": "🇺🇸",
    "Czech Republic": "🇨🇿",
    "DR Congo": "🇨🇩",
    "New Zealand": "🇳🇿",
    "Iran": "🇮🇷",
    "Egypt": "🇪🇬",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "South Africa": "🇿🇦",
    "Ghana": "🇬🇭",
    "Qatar": "🇶🇦",
    "Iraq": "🇮🇶",
    "Jordan": "🇯🇴",
    "Bosnia and Herzegovina": "🇧🇦",
    "Cura\u00e7ao": "🇨🇼",
    "Saudi Arabia": "🇸🇦",
    "Cape Verde": "🇨🇻",
    "Haiti": "🇭🇹"
}

# Function to prepend the flag to the name
def add_flag(team_name):
    flag = flag_map.get(team_name, "🏳️") # Default white flag if not found
    return f"{flag} {team_name}"

# Apply it to your dataframe BEFORE creating display_df
df["Team"] = df["Team"].apply(add_flag)

# Create results table
display_df = df[["Team"] + stages]

# Convert to HTML to preserve the <span> tags
st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

