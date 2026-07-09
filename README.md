# World Cup 2026 Predictor

Building a Monte Carlo simulation to generate outcome probabilities for this year's Men's FIFA World Cup.


## To-Do

- Merge the simulation script with the analsyis notebook to produce the processed csv file

### Repository Structure

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

