import streamlit as st 
import json 

with open("config.json","r") as f:
    config_file = json.load(f)

st.set_page_config(layout="wide", page_title="WC 2026 Predictor")

with st.sidebar:
    st.markdown(f"*Version Number: {config_file['version_num']}*")

st.title("ℹ️ About")

simple_elo_method_latext = r"""

When first starting this project, I used a simple simulation
methodology that relied entirely on a team's Elo, sourced
from `eloratings.net`. This was never intended to be a final
or robust methodology, but rather served as a simpler 
introduction to the project.

Nonetheless, since the results were not too laughable, mainly
due to the order of winning probabilities matching unsurprisingly close
with Elo (and notably Opta), I have kept this discussion of the
methodology for completeness.
            
### Group Stage
            
###### Logic
            
Due to the small group size and resulting number of matches
(4 teams, 6 matches), the full space of outcomes for a single
group can be simulated (3 outcomes ^ 6 matches = 729 possible
paths). 
            
For each of the six matches, the probabilities
of all three outcomes (home win, away win, draw) can be computed,
and the resulting combinations of all possible outcomes 
calculated. This gives a complete probability distribution for
the final state of the group (more specifically, the points
for each team). 
            
One limitation of this logic is that it does not account for
the scores of matches and the resulting goal difference. Whilst
irrelevant in some outcomes, goal difference is essential when
teams finish on equal points. Therefore, attempts should be 
made to model a team's scoring potential to mitigate this limitation.
In this simplified model, no such logic is included. Instead,
a simple method of 'breaking ties' is used to deal with such
situations (detailed below).
            
###### Implementation

Currently, the method for determining the outcome probabilities
is not very robust. 
            
ELO ratings, sourced online, are used to calculate scores for the 
2 teams: 
            
$$ Q = 10^{\text{ELO}/400} $$
            
As is standard, these are then used to determine win probabilities
for the two teams using:
            
$$ p_{a} = \frac{Q_{a}}{Q_{a}+Q_{b}} $$

However, this does not account for the possibility of a draw.
Therefore, the draw probability is calculated as:

$$ p_{\text{draw}} = \frac{1 - p_{\text{diff}}}{2} $$

where $p_{\text{diff}}$ is the absolute difference in the calculated
win probabilities. The win probabilities are then rescaled equally
to ensure the sum of probabilities is 1.

This method for calculating a draw probability is not intended to be
robust or particularly justified. Instead, it simply acts as a filler
whilst more detailed methods can be developed. The logic behind it
is the assumption that draws are more likely between equally matched
teams.

The above method allows for determining the probabilities of teams
earning different numbers of points throughout the group stage.
However, it does not account for goal difference in determining the
order of positions when on equal points. Therefore, as a simple first
method before detailed simulation, every order of teams on equal points
is considered equally likely, and the probability distributed 
accordingly.

                        
### Knockout Rounds
            
###### Logic

Unlike the group stage, the event space for the knockout rounds
is astronomocially large. 32 teams qualify for the knockout rounds,
with 31 matches having to be played to determine the winner. 
Therefore, for a single selection of 32 qualifying teams, there
are $2^{32} \approx 4.3 \times 10^{9}$ possible paths of the
kncokout round.

Additionally, there are $12 \choose 8$ ways of choosing which 8
groups the third place team qualifies from. For each of those,
there are ${4\choose3}^{8} \times {4\choose2}^{4}$ of selecting the
qualifying team. All together, this gives $\sim 4.2 \times 10^{10}$
arrangements of qualifying teams, giving a total knockout round
sample space on the order of $10^{20}$. Therefore, it is not feasible
to approach this round in the same way as the group stages.

Instead, a Monte Carlo approach is taken, where a large number of
simulations are run, and the averages can be used to estimate 
respective outcome probabilities.
            
###### Implementation

From simulating the group stage, a probability distribution for all
possible final group standings (and hence qualifying teams) has been
determined. One subtle nuance in determing the qualifying teams exists
in the third place qualifiers. These are determined by highest point 
scoring third place teams, however in instances of equal points, goal
difference is once again used. Therefore, as in the group stages,
such situations are currently split with equal probability.

To determine the qualifying teams, one outcome from each group
is sampled, weighted by its respective probability.

Once the qualifying teams have been determined, and the knockout
bracket filled, the knockout stage is simulated. To determine the
win probability of each team, the same ELO logic as above is applied.
However, since draws are no longer possible, the rescaling for
the draw probability is no longer performed.

A large number of simulations is then performed, keeping track of the
round each team reaches. The proportion of runs where a team
reaches a particular round is then used to approximate its
probability of reaching that round. The error in such an approximation
scales as $\frac{1}{\sqrt{N}}$, with $N$ being the number of iterations.
Therefore, as the number of simulatons increases, the resulting 
probabilities converge on those that would be obtained from an 
explicit handling of the entire space.
            
"""


XGBoost_method_latext = r"""
## How does the simulation work?

This page will outline the workings of the current simulation.
Note that I intend to work on and improve the simulation, and
therefore this page will be updated accordingly.

Coming soon...

            
"""



st.write(XGBoost_method_latext)

st.divider()
with st.expander("Simple Elo based Methodology"):
    st.write(simple_elo_method_latext)