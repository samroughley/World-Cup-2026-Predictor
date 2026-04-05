import streamlit as st 
import pandas as pd 
import json 


## Load Data ##

# My Predictions
with open("Monte Carlo Simulation/runs/quick_run/final_results.json","r") as f:
    simulation_results = json.load(f)

    # Rename
    for team, outcomes in simulation_results.items():
        simulation_results[team]["Final"] = simulation_results[team]["F"]
        del simulation_results[team]["F"]



# Opta's Predictions
opta_preds = pd.read_csv('misc/Opta_Predictions.csv')[['Team','R32_Exit','R16','QF','SF','Final','Winner']]
opta_preds[['R32_Exit','R16','QF','SF','Final','Winner']] = opta_preds[['R32_Exit','R16','QF','SF','Final','Winner']].astype('float')
opta_preds = opta_preds.rename(columns={"R32_Exit": "R32"})


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

        opta_prediction = opta_preds[opta_preds["Team"]==team][out].values

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
    df[stage] = df.apply(combine_probs, axis=1, args=(f"{stage}_mine", f"{stage}_Opta"))

# Custom CSS to make the bracketed text faint
st.markdown("""
    <style>
    .opta-text {
        color: #808080; /* Grey color */
        font-size: 0.85em;
        font-weight: 300;
    }
    table {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)




## Create Dashboard ##

# Create page
st.set_page_config(layout="wide",
                   page_title="WC 2026 Predictor",
                   page_icon="🏆")


st.title("🏆 2026 World Cup Probability Dashboard")

st.markdown("""
Predicted probabilities of each team reaching each knockout round in the Men's 2026 FIFA World Cup,
compared with those from Opta in brackets. \n
            
*Note: Results of -1.00% in brackets indicate missing data pulled from Opta.*
""")


# Create results table
display_df = df[["Team"] + stages]

# Convert to HTML to preserve the <span> tags
st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

