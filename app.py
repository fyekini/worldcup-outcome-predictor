"""
app.py

Streamlit app: pick two teams, get a predicted outcome probability breakdown.
Run with: streamlit run app.py
"""

import pickle
import sqlite3

import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"
DB_PATH = "worldcup.db"

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽")

st.title("⚽ FIFA World Cup 2026 Match Outcome Predictor")
st.write(
    "Select two national teams to see predicted win / draw / loss "
    "probabilities, based on a model trained on 23,000+ real international "
    "matches (2002-present), Elo-derived strength ratings, and recent form."
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_form_lookup():
    conn = sqlite3.connect(DB_PATH)
    feats = pd.read_sql(
        "SELECT * FROM team_match_features ORDER BY date", conn, parse_dates=["date"]
    )
    conn.close()
    latest = feats.groupby("team").tail(1).set_index("team")
    return latest[["form_pts_last5", "avg_goal_diff_last5", "team_rank"]]


form_lookup = load_form_lookup()
TEAMS = sorted(form_lookup.index.tolist())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home / Team A", TEAMS, index=TEAMS.index("Brazil") if "Brazil" in TEAMS else 0)
with col2:
    away_team = st.selectbox("Away / Team B", TEAMS, index=TEAMS.index("Argentina") if "Argentina" in TEAMS else 1)

if st.button("Predict Outcome"):
    if home_team == away_team:
        st.warning("Pick two different teams.")
    else:
        model = load_model()
        a = form_lookup.loc[home_team]
        b = form_lookup.loc[away_team]

        features = pd.DataFrame([{
            "rank_gap": b.team_rank - a.team_rank,
            "home_form": a.form_pts_last5,
            "away_form": b.form_pts_last5,
            "home_goal_diff_form": a.avg_goal_diff_last5,
            "away_goal_diff_form": b.avg_goal_diff_last5,
        }])

        probs = model.predict_proba(features)[0]
        away_win, draw, home_win = probs

        st.subheader("Predicted Probabilities")
        st.metric(f"{home_team} win", f"{home_win:.1%}")
        st.metric("Draw", f"{draw:.1%}")
        st.metric(f"{away_team} win", f"{away_win:.1%}")

        st.bar_chart(pd.DataFrame({
            "Outcome": [f"{home_team} win", "Draw", f"{away_team} win"],
            "Probability": [home_win, draw, away_win],
        }).set_index("Outcome"))

st.caption(
    "Built with XGBoost, scikit-learn, and Streamlit. "
    "Model trained on real historical international match results "
    "(martj42/international_results, 2002-present)."
)
