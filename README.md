# 🏆 World Cup 2026 Predictor

An end-to-end project utilising an XGboost regressor and a custom Monte Carlo simulation engine to predict the outcomes of the 2026 FIFA World Cup.

<div align="center">

<p><strong>Explore the Interactive Dashboard, containing the full simulation results and analysis, as well as a detailed technical overview of the simulation mechanics:</strong></p>

<a href="https://samroughley-world-cup-2026-predictor.streamlit.app" target="_blank">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" 
       alt="Streamlit App" 
       width="200"/>
</a>

 \
⚠️ **Important Note on Predictions & Timeline** \
While this repository may show recent commits for structural refactoring and code cleanup,
**all simulation runs and predictions were generated in April 2026**.
The model was trained strictly on historical data and was locked before the tournament kicked off to ensure authentic, unbiased predictive validity.

</div>


## 📝 Overview

This project calculates probabilities for each team to reach each knockout round in the 2026 FIFA World Cup, along with a probability of each team winning the tournament.

To do this, an XGBoost model is used within a Monte Carlo simulation, simulating the entire tournament from the group stages to the final hundreds of thousands of times. 

The XGBoost model is trained on historical data with a Poisson objective to predict the mean number of goals scored by each team in a match, taking as inputs a team's elo, along with historical scoring records. A score can then be randomly drawn from the Poisson distributions with the predicted mean rates.

After performing the simulation, the five teams with the highest predicted probability of winning the entire tournament are:

| Team | Prob. | Opta Prob. |
| --- | --- | --- |
| France | 8.2 % | 12.72 %|
| Argentina | 8.0 % | 10.31 % |
| Portugal | 6.6 % | 6.98 % |
| Spain | 6.0 % | 15.46 % |
| England | 6.0 % | 10.91 % |

Whilst these probabilities are somewhat different to those published by Opta, the same five teams are given the greatest tournament winning probability (albeit in a different order).

The full set of predictions, for all teams and knockout rounds, along with more analysis of the results and a more detailed technical breakdown of the simulation are included on [Streamlit](https://samroughley-world-cup-2026-predictor.streamlit.app)


## 🛠️ Tech Stack & Key Features

- **Machine Learning:** XGBoost Regressor for Poisson modelling of goals for match-outcome probabilities.
- **Simulation Engine:** Custom Monte Carlo simulator, dynamically adjusting relative strengths to simulate team momentum.
- **Frontend:** Streamlit dashboard for data visualisation and content hosting.
- **Data Sources:** [Historical match data](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?resource=download) from Kaggle, and pre-tournament Opta predictions


## 📊 Known Limitations & Future Work

* **Knockout Stage Home Advantage:** The current iteration of the simulation engine does not apply a home-field advantage during the knockout rounds. Because the 2026 World Cup is uniquely split across three host nations (USA, Mexico, and Canada), dynamically calculating true home advantage requires mapping branching bracket paths to exact venue locations. This may be introduced in a later, post-tournament version, investigating its effects.


## 💻 Local Installation & Usage

This repository is fully self-contained. Therefore, should you please, you can download the code and run your own simulation.

1. **Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run Simulation**

```bash
cd src
python simulation.py --run_name example_run --iterations 1000 --checkpoint_freq 100
cd ..
```

3. **View Results**

```bash
streamlit run Home.py
```


## Repository Structure

```text
World-Cup-2026-Predictor
├── data/               # Data for app/simulation
│   ├── inputs/
│   │   ├── ...
│   │   └── ...
│   ├── reference/
│   │   ├── ...
│   │   └── ...
│   └── test_run_processed_results.csv
│
├── notebooks/          # Full XGBoost training workspace
│   ├── data
│   │   ├── raw/
│   │   │   └── [Kaggle data]
│   │   └── results_with_stats.csv
│   ├── process_raw_data.ipynb
│   └── XGBoost_modelling.ipynb
│
├── pages/              # Streamlit sub-pages
│   ├── 1_Analysis.py
│   └── 2_About.py
│
├── src/                # Core Python simulation engine
│   ├── models/
│   │   └── XGBoost_regressor_model.json
│   ├── engine.py
│   └── simulation.py
│
├── .gitignore
├── config.json         # App configuration (version, etc.)
├── Home.py             # Streamlit main landing page
├── README.md
├── requirements.txt    # Python dependencies
└── LICENSE             # MIT License
```

## Personal To-Do

- [ ] Modify Streamlit dashboard to visualise the selected runs results
- [ ] Modify Streamlit analysis to only show for my run, or hide text when using a different run
- [ ] Finish technical overview on Streamlit
- [ ] Map knockout venues to dynamically apply a home advantage during the knockout round simulations
