import streamlit as st 
import pandas as pd 
import plotly.graph_objects as go
import json
import numpy as np

# Create page
st.set_page_config(layout="wide",
                   page_title="Analysis",
                   page_icon="📈")


## Load Data ##

with open("config.json","r") as f:
    config_file = json.load(f)

# Define the name of the run (boilerplate)
RUN_NAME = "test_run"

# My Predictions
# with open("Monte Carlo Simulation/runs/quick_run/final_results.json","r") as f:
with open("data/outputs/test_run/final_results.json","r") as f:
    simulation_results = json.load(f)
    simulation_results.pop('conduct_skip_counter')

    # Rename
    for team, outcomes in simulation_results.items():
        simulation_results[team]["Final"] = simulation_results[team]["F"]
        del simulation_results[team]["F"]


# Opta's Predictions
opta_preds = pd.read_csv('data/reference/Opta_Predictions.csv')[['Team','R16','QF','SF','Final','Winner']]
opta_preds[['R16','QF','SF','Final','Winner']] = opta_preds[['R16','QF','SF','Final','Winner']].astype('float')
opta_preds = opta_preds.rename(columns={"R32_Exit": "R32"})

# Processed Data
processed_data_df = pd.read_csv(f'data/outputs/{RUN_NAME}/processed_results.csv')


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

with st.sidebar:
    st.markdown(f"*Version Number: {config_file['version_num']}*")


st.title("📈 Analysis")

st.markdown("""
This page contains a more comprehensive look at the predictions made by
the simulation. This includes comparisons with the predictions published
by Opta, which are taken as the most accuracte prediction probabilities
for this task.
            
##### Simple Comparison
            
The simulation outputs probabilites for every team reaching each of the 
knockout rounds, as well as a probability of winning the entire 
competition. Opta, a well established organisation, also publishes its
own predictions, likely a result of a much more comprehensive study. These
predictions include probabilities for each team reaching each of the knockout
rounds from the round of 16 onwards, as well as probabilities of becoming 
the champion. Therefore, these two sets of probabilities can be directly 
compared.
""")







# Create figure 

y_axis_ranges = {
    "Final": 20,
    "Winner": 10,
    "R16": 80,
    "QF": 50,
    "SF": 35
}

# Create the figures
figs = []
for stage in stages:

    # Scatter
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

    # trendline
    coeffs = np.polyfit(df[f"{stage}_Opta"],df[f"{stage}_mine"],2)
    polyfunc = np.poly1d(coeffs)
    x_range = np.linspace(df[f"{stage}_Opta"].min(),df[f"{stage}_Opta"].max(),100)
    y_fit = polyfunc(x_range)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_fit,
        name='Model Trend',
        line=dict(color='rgba(255, 0, 0, 0.4)', width=2)
    ))

    # y=x line
    max_prob = min(df[f"{stage}_Opta"].max(), df[f"{stage}_mine"].max())
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
            y=0.98
        ),
        xaxis=dict(
            title=dict(
                text="Opta"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Mine"
            ),
            range=[0,y_axis_ranges[stage]]
        ),
        margin=dict(l=20, r=20, t=40, b=40),
        height=350,
        autosize=False
    )
    figs.append(fig)

for fig in figs:
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

with st.container(border=True):
    # Main figures
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figs[3])
    with col2:
        st.plotly_chart(figs[4])

    # Other figures
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(figs[0])
    with col2:
        st.plotly_chart(figs[1])
    with col3:
        st.plotly_chart(figs[2])

    st.markdown("""
    *Comparison of the predicted probabilities of each team reaching 
    each round within the tournament with those calculated by Opta.
    The light grey line represents equal probabilities from each 
    source (y=x line). The red line is a quadratic curve fit to the data.
    There are clear disagreements between my simulation and the predictions
    published by Opta, including a different ordering.*
    """)






st.markdown("""
From the above comparion, it is clear that there is some
consistency between the two sets of predictions, however 
there are also some stark divergences.
            
For the round of 16, the data has a reasonable agreement with 
those from Opta (in the context of expectations). Observing
the fitted quadratic, only a subtle convex curve is visible.

However, as we move deeper into the competition, the curve
becomes increasingly convex. The shape of this curve suggests
suggests that my simulation gives the strongest teams a lower 
probability of progressing than Opta, with the gradient
slightly above the 1 in the bottom left representing
a higher probability given to 'upsets'.
            
Another notable feature, also highlighted by the quadratic 
shape, is how the two simulations do not always agree on
which teams have the stronger probability. For example, whilst
Opta predict Spain as favourites to win the entire tournament,
my simualation places Spain as fifth favourites. This is rather 
unexpected, as Spain are the highest Elo team, with Elo being 
one of the features used by the model. 
            
There is obviously no reason for the order to align perfectly
with Elo rankings. For one, the simulation uses information 
other than just Elo (see the About section). Additionally,
the simulation also accounts for the paths teams can take to
the final, and hence some changes in ranking may even indicate
an effective simulation.
            
Whilst a true understanding of the observations above would
likely require analysing the model's decision making
process (which I may do in later versions), the order of
rankings is discussed more in the following section.
            

##### Elo
            
Elo ratings are often thought of as some of the best measures
of a team's relative strength. When I first developed this 
simulation, I used a method that relied entirely on Elo and
managed to achieve fairly believable results. 
            
However, it is a given that using only Elo ratings will 
drastically oversimplify the problem. It is less obvious
how strong Elo rankings on in determining which team
has the higher probability of, for example, winning the 
competition. Whilst Elo will miss more nuanced questions
about how a team plays, it will more significantly not be
able to account for the tournament draw.
            
As is always the case, some teams will have to face
more challenging opponents in their journey to the final,
and there is some determinism in how this may unfold.
Therefore, this has the potential to shuffle team 
rankings between Elo and win probabilities. As a result, 
figure below shows how the different rankings compare.
""")




# Create figure
processed_data_df['Elo_Rank'] = processed_data_df['Elo'].rank(ascending=False, method='min')
processed_data_df['Opta_Rank'] = processed_data_df['Winner_Opta'].rank(ascending=False, method='min')
processed_data_df['My_Rank'] = processed_data_df['Winner_mine'].rank(ascending=False, method='min')

# processed_data_df = processed_data_df[processed_data_df['My_Rank']<30]

# Build the slope chart
fig = go.Figure()

# Define the order of the columns
cols = ['Opta Rank', 'Elo Rank', 'My Rank']

for i, row in processed_data_df.iterrows():

    # Highlight teams where model differs from Elo significantly
    diff = row['My_Rank'] - row['Elo_Rank']
    if diff > 0:
        color, width = "red", 1
    elif diff < -0:
        color, width = "green", 1
    else:
        color, width = "lightgrey", 1
    
    fig.add_trace(go.Scatter(
        x=cols,
        y=[row['Opta_Rank'],row['Elo_Rank'],row['My_Rank']],
        mode='lines+markers',
        name=row['Team'],
        line=dict(color=color, width=width),
        hovertemplate=f"<b>{row['Team']}</b><br>Opta: %{{y}}<br>Elo: {row['Elo_Rank']}<br>Mine: {row['My_Rank']}"
    ))

fig.update_layout(
    title="How Models Rank the Teams",
    yaxis=dict(autorange="reversed", title="Rank (1st at Top)"),
    showlegend=False,
    height=400
)

st.plotly_chart(fig, use_container_width=True)





st.markdown("""
Neither Opta or my own simulation agree completely with Elo
rankings. Whilst this is not a casue for concern, it is an 
interesting observation...
""")