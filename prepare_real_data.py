"""
prepare_real_data.py

Converts the real international match results dataset (martj42/international_results,
mirrored from Kaggle: "International football results from 1872 to present") into the
schema used by data_pipeline.py.

The raw dataset has no FIFA ranking column, so this script computes a rolling Elo
rating for every team from match history -- a stronger, self-updating signal than
a static rank anyway, and a good talking point in an interview ("I built my own
Elo system instead of relying on a static ranking table").

Run this once, then point data_pipeline.py at the output.
"""

import pandas as pd

RAW_PATH = "data/real_matches_raw.csv"
OUT_PATH = "data/matches.csv"

K_FACTOR = 20
START_ELO = 1500


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo(df):
    elo = {}

    def get_rating(team):
        return elo.get(team, START_ELO)

    home_elo_before = []
    away_elo_before = []

    for _, row in df.iterrows():
        home, away = row.home_team, row.away_team
        r_home, r_away = get_rating(home), get_rating(away)
        home_elo_before.append(r_home)
        away_elo_before.append(r_away)

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score == row.away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        exp_home = expected_score(r_home, r_away)
        elo[home] = r_home + K_FACTOR * (actual_home - exp_home)
        elo[away] = r_away + K_FACTOR * ((1 - actual_home) - (1 - exp_home))

    df["home_elo"] = home_elo_before
    df["away_elo"] = away_elo_before
    return df


def main(min_year=2002):
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Raw dataset: {len(df):,} matches from {df.date.min().date()} to {df.date.max().date()}")

    # Elo needs full history to be accurate, so compute it on the full dataset first
    df = compute_elo(df)

    # Then trim to recent matches for training -- modern football plays differently
    # than 1870s football, and a smaller, more relevant window trains faster
    df = df[df.date.dt.year >= min_year].copy()

    # Convert Elo to a "rank-like" number (lower = better, matching FIFA rank convention)
    # by ranking teams within each year
    df["year"] = df.date.dt.year
    df["home_fifa_rank"] = df.groupby("year")["home_elo"].rank(ascending=False, method="first")
    df["away_fifa_rank"] = df.groupby("year")["away_elo"].rank(ascending=False, method="first")

    out = df.rename(columns={
        "home_score": "home_goals",
        "away_score": "away_goals",
    })[[
        "date", "home_team", "away_team", "home_goals", "away_goals",
        "home_fifa_rank", "away_fifa_rank", "tournament"
    ]]

    out.to_csv(OUT_PATH, index=False)
    print(f"Filtered to {len(out):,} matches ({min_year}-present) -> {OUT_PATH}")
    print("Elo-derived rankings written as home_fifa_rank / away_fifa_rank.")


if __name__ == "__main__":
    main()
