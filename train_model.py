"""
train_model.py

Trains an XGBoost classifier to predict match outcome (home win / draw / away win)
using engineered features from data_pipeline.py. Saves the trained model and
prints accuracy + log-loss for your resume bullet.
"""

import sqlite3
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb

DB_PATH = "worldcup.db"
MODEL_PATH = "model.pkl"


def build_match_level_dataset(conn):
    """Rejoin the long-format team features back into one row per match."""
    matches = pd.read_sql("SELECT * FROM matches", conn, parse_dates=["date"])
    feats = pd.read_sql("SELECT * FROM team_match_features", conn, parse_dates=["date"])

    home_feats = feats[feats.is_home == 1][
        ["date", "team", "form_pts_last5", "avg_goal_diff_last5"]
    ].rename(columns={
        "team": "home_team",
        "form_pts_last5": "home_form",
        "avg_goal_diff_last5": "home_goal_diff_form",
    })
    away_feats = feats[feats.is_home == 0][
        ["date", "team", "form_pts_last5", "avg_goal_diff_last5"]
    ].rename(columns={
        "team": "away_team",
        "form_pts_last5": "away_form",
        "avg_goal_diff_last5": "away_goal_diff_form",
    })

    merged = matches.merge(home_feats, on=["date", "home_team"], how="left")
    merged = merged.merge(away_feats, on=["date", "away_team"], how="left")

    merged["rank_gap"] = merged["away_fifa_rank"] - merged["home_fifa_rank"]
    merged["home_form"] = merged["home_form"].fillna(1.0)
    merged["away_form"] = merged["away_form"].fillna(1.0)
    merged["home_goal_diff_form"] = merged["home_goal_diff_form"].fillna(0.0)
    merged["away_goal_diff_form"] = merged["away_goal_diff_form"].fillna(0.0)

    def outcome(row):
        if row.home_goals > row.away_goals:
            return 2  # home win
        elif row.home_goals == row.away_goals:
            return 1  # draw
        else:
            return 0  # away win

    merged["result"] = merged.apply(outcome, axis=1)
    return merged


FEATURES = [
    "rank_gap", "home_form", "away_form",
    "home_goal_diff_form", "away_goal_diff_form",
]


def main():
    conn = sqlite3.connect(DB_PATH)
    df = build_match_level_dataset(conn)
    conn.close()

    X = df[FEATURES]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)

    print(f"Test accuracy: {acc:.3f}")
    print(f"Test log-loss: {ll:.3f}")
    print("\nClassification report (0=away win, 1=draw, 2=home win):")
    print(classification_report(y_test, preds))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {MODEL_PATH}")
    print("Use this accuracy figure in your resume bullet once trained on real data.")


if __name__ == "__main__":
    main()
