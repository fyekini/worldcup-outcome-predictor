"""
data_pipeline.py

Loads raw match data into a SQLite database and builds engineered features:
- rolling form - points from last 5 matches
- goal differential average
- FIFA rank gap between teams

"""

import sqlite3
import pandas as pd

DB_PATH = "worldcup.db"
CSV_PATH = "data/matches.csv"


def load_raw_data(conn):
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df.to_sql("matches", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into 'matches' table")
    return df


def build_team_match_log(df):
    """Reshape home/away rows into one row per team per match (long format)."""
    home = df.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "goals_for", "away_goals": "goals_against",
        "home_fifa_rank": "team_rank", "away_fifa_rank": "opponent_rank",
    })
    home["is_home"] = 1

    away = df.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "goals_for", "home_goals": "goals_against",
        "away_fifa_rank": "team_rank", "home_fifa_rank": "opponent_rank",
    })
    away["is_home"] = 0

    cols = ["date", "team", "opponent", "goals_for", "goals_against",
            "team_rank", "opponent_rank", "is_home", "tournament"]
    long_df = pd.concat([home[cols], away[cols]], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"])

    long_df["points"] = long_df.apply(
        lambda r: 3 if r.goals_for > r.goals_against
        else (1 if r.goals_for == r.goals_against else 0), axis=1
    )
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]

    # Rolling form over last 5 matches per team (shifted to avoid leakage)
    long_df["form_pts_last5"] = (
        long_df.groupby("team")["points"]
        .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )
    long_df["avg_goal_diff_last5"] = (
        long_df.groupby("team")["goal_diff"]
        .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )

    return long_df


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_raw_data(conn)
    features = build_team_match_log(df)
    features.to_sql("team_match_features", conn, if_exists="replace", index=False)
    print(f"Built feature table with {len(features)} rows -> 'team_match_features'")
    conn.close()


if __name__ == "__main__":
    main()
