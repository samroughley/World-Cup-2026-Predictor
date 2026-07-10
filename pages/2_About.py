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

This page contains a more detailed overview of the simulaton engine, covering both the
XGBoost model (training and input features), and the construction of the Monte Carlo
engine.

### XGBoost Model

In order to simulate the tournament, we first must be able to pick an outcome for each
individual match. There are an incredibly wide range of possible techniques to do this,
each with their own strengths and weaknesses. For this simulation, an XGBoost model was used
due to the algorithm's strong performance on tabular data and ability to be trained effectively
on smaller datasets.


#### Objective

Before training the model, we must determine its objective. It is natural to instinctively
think of training the model to perform as a classifier, assigning probabilities to each 
possibility: home win, away win, draw (not possible in knockout rounds). However, such an
approach has limitations when applied to our simulation.

Whilst the winner of a match is definitely the most significant result, the final scoreline
is also incredibly important. When teams finish on the same number of points in a group, FIFA
outlines procedures to determine which team ranks higher. This same tiebreaking requirement
occurs when determining which of the third place teams qualify for the knockout rounds. Whilst
these tiebreaking rules do typically look at points scored in relevant matches first, should teams
not be separated by these rules, goal difference is often the next check. Therefore, in order
to effectively simulate the tournament, it is vital we have a way of simulating exact scorelines.

As such, an XGBoost classifier is not sufficient. Instead, we take a different approach, using an
XGBoost regressor model to predict specific scorelines. To do this, we make use of a native feature
in the xgboost package: **Poisson modelling**. 

```python
# Create the model
model = xgb.XGBRegressor(
    objective='count:poisson',  # the key feature

    # Standard hyperparameters (starting point)
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,

    # Help prevent overfitting
    subsample=0.8,
    colsample_bytree=0.8,

    # Evaluation metric
    eval_metric='poisson-nloglik',
    random_state=42
)
```

By using this objective, we tell the model that our target variable represents the number of occurrences
of an event that follows a Poisson distribution. Whilst not the most robust justification, the choice
of a Poisson distribution can be understood as satisfying the fact that goals occur in discrete
amounts, the tell-tale behaviour of Poisson distributions.

As a result, when training the model, we are really teaching the model to predict the mean rate of
goals scored by a particular team in a particular match. This mean rate uniquely defines the 
corresponding Poisson distribution, from which we can sample a specific scoreline.

In sampling a scoreline for the game, we run inference with the model twice, determining the
mean goal rates for the home and away teams respectively. These then define two distributions, each
drawn from individually to sample a scoreline. Whilst this can give believable results, it does make
a likely incorrect assumption that the numbers of goals scored by each team are independent, since
teams will adjust their strategies depending upon the current scores. However, this effect was
determined to be marginal with regards to the final predicted probabilities for the tournament.

#### Training

To train the model, we first need data to train on. Thankfully, a dataset of historical international
football results dating back to 1872 is hosted on [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).

Now that we have the data, we need to determine what features we wish to pass to the model. For our
simulation, we use the Elo ratings of the two teams playing, a selection of EMAs, and a home
advantage indicator.

| Feature | Description | 
| --- | --- | 
| Elo | Dynamic Elo rating for each team, following the same framework as `eloratings.net` |
| goals scored EMA short | Fast EMA of the goals each team has scored in previous matches |
| goals conceded EMA short | Fast EMA of the goals each team has conceded in previous matches |
| goals scored EMA long | Slow EMA of the goals each team has scored in previous matches |
| goals conceded EMA long | Slow EMA of the goals each team has conceded in previous matches |
| home advantage | A single indicator for whether either team has a home advantage |

The above metrics were calculated for all the historical data, with no future leakage. Before
training, the data was filtered to only include matches where teams had at least 30 historical
matches to allow the metrics to warmup. The model was trained to predict the number of 
goals the home team scored. Therefore, the data was duplicated and mirrored prior to training.
The choice of this approach, rather than predicting both home and away distributions, was done
to mitigate any home advantage potentially being learned through column names, instead having
to be identified through the explicit home advantage column.

For an analysis of the model's predictive performance, including comparisons against baseline
classifiersm see the Model Analysis page.


### Tournament Simulation

Now that we have a trained XGBoost model that can be used to sample scorelines for individual matches,
we need to build the actual Monte Carlo simulation engine. A lot of this implementation is fairly trivial,
and therefore only a brief overview is contained below.

#### Group Stage

Simulating the group stages is fairly simple. For each match, we sample a scoreline using our trained
model, determine which team won (or if it was a draw), and update the group table with the new points
tally and goals scored and conceded for each team.

One feature worth noting is how the model's input features are dynamically adjusted throughout the
tournament. It was discussed above how the model takes as inputs a team's elo rating, along with 
moving averages of the number of goals they have scored and conceded. These are not static measures,
but instead will change throughout the tournament. Therefore, along with updating the group tables,
these features for each team are updated throughout the simulation, as though the scores we sample
actually materialise. This puts the simulation much closer to reality, allowing it to account for
a team's momentum throughout the tournament. It is therefore important that the matches are simulated
in the order they are actually played.

Once the group stage has been simulated, we have the full set of group tables and must determine which
teams qualify for the knockout rounds, including handling teams on equal points and the qualification of
the eight best third place teams. Thankfully, FIFA outlines an exact procedure for this determination
prior to the tournament, including who plays who in the knockout rounds. In order to apply these
procedures, we must keep track of the scorelines of each individual match (along with the overall group
tables), after which we simply just follow the rules in order.



#### Knockout Stage

Simulating the knockout rounds is potentially simpler than the group stages, due to not needing to
record anything from previous matches other than the winner (e.g. no third place team tiebreaking
involved). However, it does have its own challenges.

Namely, there must be a winner in every match, leading to extra time and penalties when required. 
Our model was trained to predict scorelines, but does not have an ability to predict the winner
in a penalty shootout. Therefore, we employ some tricks to handle these cases.

Firstly, when a match goes to extra time, we use the same Poisson distributions, however with the
mean rate divided by three (to reflect only 30 minutes of game time). We then sample a specific
scoreline to determine if a team wins in extra time, or if the match goes to penalties. This
method is a rough workaround, since it doesn't account for different tactical approaches
in extra time or player fatigue, however should still account for the general relative strengths
and weaknesses of the two teams.

It is also worth noting a structural constraint within our training data. Whilst our simulation
engine uses the model to predict scorelines strictly within regulation 90 minutes, the historical
Kaggle dataset includes final scorelines after extra time (when it is played). The source data does
not explicitly flag whether a match went to extra time, or provide the 90 minute regulation scoreline
for those specific games. This introduces a minor, unavoidable inconsistency between our training data
and model inference. However, given that extra time is only played in a tiny fraction of total
international fixtures historically, this data limitation was determined to have a negligible impact
on the model's baseline scoring rates and was accepted as a reasonable trade-off.

If penalties are reached, we employ another rough workaround to determine the winner. We randomly
select a winner from the two teams, with probabilities weighted by their expected goals (Poisson mean
rate) predicted by the XGBoost model. Whilst we would typically expect the team with the higher 
expected goals to be the overall stronger team, and hence have the higher chance in the penalty
shootout, penalties are notorious for giving underdogs the best chance of qualification due
to their unpredictability and distinction from open play. None of this is explicitly reflected in our
workaround.

            
"""



st.write(XGBoost_method_latext)

st.divider()

st.write(r"""
#### Archive
         
Before building the simulation using the trained XGBoost model, I built
the simulation framework using a method that relied solely on a team's elo
to determine win probabilities. This framework was completely overhauled
in generating the simulation's final results.
         
This simulation framework was incredibly rudimentary, filled with cheap
workarounds to keep it as simple as possible. However, it's approach did
differ slightly from that described above, for example with the entire group
stage event space being mapped out and drawn from for the knockout stage.

Therefore, simply for completeness and archiving, I have included an overview
of this methodology below.
""")
with st.expander("Simple Elo based Methodology"):
    st.write(simple_elo_method_latext)