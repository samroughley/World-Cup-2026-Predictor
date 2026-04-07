import streamlit as st 
import pandas as pd 
import plotly.graph_objects as go
import json
import numpy as np

## Load Data ##

with open("config.json","r") as f:
    config_file = json.load(f)

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

stages = ["R16","QF","SF","Final","Winner"]
for team, outcomes in simulation_results.items():
    data["Team"].append(team)

    for out, prob in outcomes.items():
        if f"{out}_mine" not in data:
            continue
        data[f"{out}_mine"].append(prob*100)

        opta_prediction = opta_preds[opta_preds["Team"]==team][out].values

        if len(opta_prediction) > 0:
            data[f"{out}_Opta"].append(opta_prediction[0])
        else:
            data[f"{out}_Opta"].append(-1)



df = pd.DataFrame(data)




## Create Dashboard ##

# Create page
st.set_page_config(layout="wide",
                   page_title="Simulation",
                   page_icon="📈")

with st.sidebar:
    st.markdown(f"*Version Number: {config_file['version_num']}*")


st.title("📈 Simulation")

st.markdown("""
A more detailed exploration of the results from the simulation, showing
both comparisons of the results, as well as the dynamics of the simulation
(unfinished).
""")

st.markdown("### Probabilities Comparisons")

figs = []

for stage in stages:

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[f"{stage}_Opta"],
            y=df[f"{stage}_mine"],
            mode="markers",
            name='Predictions',
            # hovertemplate =
            #     '<i>Price</i>: $%{y:.2f}'+
            #     '<br><b>X</b>: %{x}<br>'+
            #     '<b>%{text}</b>',
            customdata=np.stack((df["Team"],), axis=-1),
            hovertemplate=
                '<b>%{customdata[0]}</b>'+
                '<br><b>Opta</b>: %{x}%<br>'+
                '<b>Simulation</b>: %{y}%'
        )
    )

    max_prob = max(df[f"{stage}_Opta"].max(), df[f"{stage}_mine"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_prob],
            y=[0, max_prob],
            name='Equivalence',
            line=dict(
                width=1,
                dash='dot',
                color='rgba(128,128,128,0.5)',
            ),
            mode='lines'
        )
    )
    fig.update_layout(
        showlegend=False,
        title=dict(
            text=stage,
            font=dict(size=20),
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.84
        ),
        xaxis=dict(
            title=dict(
                text="Opta"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Mine"
            )
        )
    )
    figs.append(fig)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.plotly_chart(figs[0])
with col2:
    st.plotly_chart(figs[1])
with col3:
    st.plotly_chart(figs[2])
with col4:
    st.plotly_chart(figs[3])
with col5:
    st.plotly_chart(figs[4])

st.markdown("""
*Comparison of the predicted probabilities of each team reaching 
each round within the tournament with those calculated by Opta.
The light grey line represents equal probabilities from each 
source (y=x line).*
""")