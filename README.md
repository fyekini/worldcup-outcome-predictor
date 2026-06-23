# worldcup-outcome-predictor

**Predicting FIFA World Cup 2026 match outcomes with XGBoost, Elo ratings, and Monte Carlo simulation.**

Trained on 23,000+ real international football matches (2002–present), this project
predicts match outcomes (win / draw / loss) and simulates the full 2026 World Cup
knockout bracket 10,000 times to forecast each team's odds of advancing and winning
the tournament.

## Results

- **57.3% test accuracy** / **0.914 log-loss** on a 3-class outcome prediction
  (home win / draw / away win), evaluated on a held-out 20% test split
- Simulated knockout bracket (10,000 runs) currently favors **Argentina (17.3%)**,
  **Brazil (15.0%)**, and **Spain (10.6%)** to win the tournament, consistent with
  pre-tournament pundit expectations — a useful sanity check on the model
- Full results: [`simulation_results.csv`](simulation_results.csv)

## How it works

1. **Data** — Real match data sourced from [martj42/international_results](https://github.com/martj42/international_results)
   (mirrored from Kaggle's "International football results from 1872 to present"),
   filtered to 2002–present for modern relevance.
2. **Elo ratings** — Since the raw dataset has no FIFA ranking column, I built a
   custom Elo rating system from scratch, updating team strength after every
   match in the dataset's full history. This was a deliberate choice over a
   static rank table since Elo self-corrects based on actual results.
3. **Feature engineering (SQL + pandas)** — Rolling 5-match form and goal
   differential per team, computed via a SQLite pipeline.
4. **Model** — XGBoost multiclass classifier predicting win/draw/loss probabilities.
5. **Simulation** — Monte Carlo simulation of the Round of 16 through the Final,
   run 10,000 times to produce probability-weighted advancement odds for every team.
6. **App** — A Streamlit interface where anyone can pick two teams and get instant
   predicted probabilities, with no Python or modeling background required.

## Project structure

```
worldcup-outcome-predictor/
├── data/
│   └── matches.csv              # real match data, processed and ready to use
├── prepare_real_data.py         # converts raw match data + computes Elo ratings
├── generate_sample_data.py      # synthetic data generator (fallback/demo only)
├── data_pipeline.py             # SQLite schema + rolling feature engineering
├── train_model.py               # XGBoost training + evaluation
├── simulate_tournament.py       # Monte Carlo bracket simulation
├── app.py                       # Streamlit interactive predictor
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Build the SQLite database and engineered features
python data_pipeline.py

# 2. Train the model
python train_model.py

# 3. Run the tournament simulation
python simulate_tournament.py

# 4. Launch the interactive app
streamlit run app.py
```

`data/matches.csv` is already populated with real, processed match data — steps
1–4 work immediately. To refresh with the latest results, re-download the source
CSV from the [martj42/international_results](https://github.com/martj42/international_results)
repo and rerun `prepare_real_data.py` first.

## Tech stack

Python · pandas · scikit-learn · XGBoost · SQLite · Streamlit

## Possible extensions

- Replace the simplified knockout bracket with the official locked bracket once
  the group stage concludes
- Add player-level injury/availability data as a feature
- Compare XGBoost against a simpler Elo-only baseline to quantify model lift
- Swap the static Round of 16 pairing logic for a full group-stage simulator
