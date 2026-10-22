"""Apple next-day closing price prediction with Linear Regression.

Run from the repository root:
    python src/stock_price_prediction.py
Expects the Kaggle file at data/Apple_stock_data.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/Apple_stock_data.csv")
IMAGE_DIR = Path("images")
FEATURES = ["Open", "High", "Low", "Close", "Volume"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["Date"] + FEATURES]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate dates:", df["Date"].duplicated().sum())
    print("Days with zero volume:", (df["Volume"] == 0).sum())
    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = next trading day's Close; the last row has no next day, so drop it."""
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)
    return df.dropna()


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def plot_history(df: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(df["Date"], df["Close"])
    axes[0].set_title("Closing price (split-adjusted)")
    axes[0].set_ylabel("USD")
    axes[1].plot(df["Date"], df["Volume"] / 1e6)
    axes[1].set_title("Trading volume")
    axes[1].set_ylabel("Millions of shares")
    fig.savefig(IMAGE_DIR / "price_and_volume_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(dates, y_test, pred) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(dates, y_test.values, label="Actual")
    axes[0].plot(dates, pred, label="Predicted")
    axes[0].set_title("Full test period")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[1].plot(dates[-100:], y_test.values[-100:], label="Actual")
    axes[1].plot(dates[-100:], pred[-100:], label="Predicted")
    axes[1].set_title("Last 100 trading days (zoomed in)")
    axes[1].set_ylabel("USD")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    plot_history(df)
    df = make_target(df)

    x, y = df[FEATURES], df["Target"]

    # Time-ordered split: first 80% of days train, last 20% test (no shuffling)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )
    n = len(x_train)
    print(x_train.shape, x_test.shape)
    print("Train:", df["Date"].iloc[0].date(), "to", df["Date"].iloc[n - 1].date())
    print("Test: ", df["Date"].iloc[n].date(), "to", df["Date"].iloc[-1].date())
    print("Train price range:", round(y_train.min(), 2), "to", round(y_train.max(), 2))
    print("Test price range: ", round(y_test.min(), 2), "to", round(y_test.max(), 2))

    # Scale using the training data only
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train
    lr = LinearRegression()
    lr.fit(x_train_scaled, y_train)

    # Evaluate against a naive baseline ("tomorrow's close = today's close")
    naive = metrics(y_test, x_test["Close"])
    pred = lr.predict(x_test_scaled)
    model = metrics(y_test, pred)
    results = pd.DataFrame({"Naive Baseline": naive, "Linear Regression": model}).round(4)
    print(results)
    print("Training R2:", round(lr.score(x_train_scaled, y_train), 4))
    print("Average test price:", round(y_test.mean(), 2))

    plot_predictions(df["Date"].iloc[n:].reset_index(drop=True), y_test, pred)


if __name__ == "__main__":
    main()
