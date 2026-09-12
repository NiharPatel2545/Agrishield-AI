"""CLI: rebuild data/agrishield_training.csv from LUCAS + WoSIS."""

from agrishield.dataset import build_training_csv


if __name__ == "__main__":
    table = build_training_csv()
    print(table["source"].value_counts().to_string())
    print("rows:", len(table))
