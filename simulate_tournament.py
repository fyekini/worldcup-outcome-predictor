"""
simulate_tournament.py

Runs a Monte Carlo simulation of a World Cup-style knockout bracket using the
trained model's win probabilities. Simulates the bracket 10,000 times and
reports each team's probability of reaching each round.

Demonstrates: probabilistic simulation, bracket logic, turning a pairwise
match model into a full-tournament forecast.
"""

import pickle
import random
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

MODEL_PATH = "model.pkl"
DB_PATH = "worldcup.db"
N_SIMULATIONS = 10000

# Real Round of 16 -- example matchups drawn from the actual 2026 World Cup
# group-stage field. Update these pairings once the official bracket locks,
# using the same team name spellings found in data/matches.csv.
BRACKET = [
    ("Brazil", "Switzerland"),
    ("Argentina", "Australia"),
    ("France", "Norway"),
    ("England", "Panama"),
    ("Spain", "Morocco"),
    ("Netherlands", "United States"),
    ("Portugal", "Colombia"),
    ("Germany", "Japan"),
]


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_latest_form(conn):
    """Pulls each team's most recent rolling form/rank values from the feature table."""
    feats = pd.read_sql(
        "SELECT * FROM team_match_features ORDER BY date", conn, parse_dates=["date"]
    )
    latest = feats.groupby("team").tail(1).set_index("team")
    return latest[["form_pts_last5", "avg_goal_diff_last5", "team_rank"]]


def predict_win_prob(model, team_a, team_b, form_lookup):
    """Returns P(team_a wins), treating team_a as 'home' for model purposes."""
    default = {"form_pts_last5": 1.5, "avg_goal_diff_last5": 0.0, "team_rank": 100}
    a = form_lookup.loc[team_a].to_dict() if team_a in form_lookup.index else default
    b = form_lookup.loc[team_b].to_dict() if team_b in form_lookup.index else default

    features = pd.DataFrame([{
        "rank_gap": b["team_rank"] - a["team_rank"],
        "home_form": a["form_pts_last5"],
        "away_form": b["form_pts_last5"],
        "home_goal_diff_form": a["avg_goal_diff_last5"],
        "away_goal_diff_form": b["avg_goal_diff_last5"],
    }])
    probs = model.predict_proba(features)[0]  # [away_win, draw, home_win]
    away_win, draw, home_win = probs
    # Split the draw probability via penalty-shootout coin flip for knockout rounds
    p_team_a = home_win + draw * 0.5
    return p_team_a


def simulate_round(matchups, model, form_lookup):
    winners = []
    for team_a, team_b in matchups:
        p_a = predict_win_prob(model, team_a, team_b, form_lookup)
        winner = team_a if random.random() < p_a else team_b
        winners.append(winner)
    return winners


def run_simulations(model, form_lookup, n_sims=N_SIMULATIONS):
    progression = defaultdict(lambda: defaultdict(int))
    all_teams = [t for pair in BRACKET for t in pair]

    for _ in range(n_sims):
        round_of_16 = list(BRACKET)
        for team in all_teams:
            progression[team]["Round of 16"] += 1

        quarterfinalists = simulate_round(round_of_16, model, form_lookup)
        for team in quarterfinalists:
            progression[team]["Quarterfinals"] += 1

        qf_pairs = list(zip(quarterfinalists[::2], quarterfinalists[1::2]))
        semifinalists = simulate_round(qf_pairs, model, form_lookup)
        for team in semifinalists:
            progression[team]["Semifinals"] += 1

        sf_pairs = list(zip(semifinalists[::2], semifinalists[1::2]))
        finalists = simulate_round(sf_pairs, model, form_lookup)
        for team in finalists:
            progression[team]["Final"] += 1

        champion = simulate_round([tuple(finalists)], model, form_lookup)[0]
        progression[champion]["Champion"] += 1

    return progression, n_sims


def main():
    model = load_model()
    conn = sqlite3.connect(DB_PATH)
    form_lookup = load_latest_form(conn)
    conn.close()

    progression, n_sims = run_simulations(model, form_lookup)

    rows = []
    for team, rounds in progression.items():
        rows.append({
            "team": team,
            "reach_QF_%": round(100 * rounds.get("Quarterfinals", 0) / n_sims, 1),
            "reach_SF_%": round(100 * rounds.get("Semifinals", 0) / n_sims, 1),
            "reach_Final_%": round(100 * rounds.get("Final", 0) / n_sims, 1),
            "win_title_%": round(100 * rounds.get("Champion", 0) / n_sims, 1),
        })

    results = pd.DataFrame(rows).sort_values("win_title_%", ascending=False)
    print(f"Monte Carlo simulation results over {n_sims:,} runs:\n")
    print(results.to_string(index=False))
    results.to_csv("simulation_results.csv", index=False)
    print("\nSaved -> simulation_results.csv")


if __name__ == "__main__":
    main()
