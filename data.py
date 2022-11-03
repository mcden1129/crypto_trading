import os
from pathlib import Path

import pandas as pd

__all__ = (
    "generate_meta_data",
    "generate_coins_data",
)

WORKING_PATH = r"C:\Dennis\crypto"
DATA_FILES_PATH = Path(WORKING_PATH + r"\archive")


def generate_meta_data():
    meta_data = []
    for f in os.scandir(DATA_FILES_PATH):
        if f.name.endswith(".csv") and f.name != "code.csv":
            ticker_1, ticker_2 = None, None
            if "-" in f.name:
                tickers = f.name.replace(".csv", "").split("-")
                ticker_1 = tickers[0]
                ticker_2 = tickers[1]
            elif len(f.name) == 10:
                ticker_1 = f.name[0:3]
                ticker_2 = f.name[3:6]

            data_df = pd.read_csv(f)
            data_df["time"] = pd.to_datetime(data_df["time"], unit="ms")
            data_df = data_df.set_index("time")

            if data_df.shape[0] > 0:
                start = data_df.index[0]
                end = data_df.index[-1]
                n_row = data_df.shape[0]
                cols = [col for col in data_df.columns]
                meta_data.append(
                    [
                        f.name,
                        ticker_1,
                        ticker_2,
                        start,
                        end,
                        n_row,
                        cols,
                    ]
                )

    meta_data_df = pd.DataFrame(
        meta_data,
        columns=[
            "filename",
            "ticker_1",
            "ticker_2",
            "start",
            "end",
            "n_row",
            "columns",
        ],
    )
    meta_data_df = meta_data_df.sort_values("start")
    return meta_data_df


def generate_coins_data():
    meta_data_df = pd.read_csv(Path(WORKING_PATH + r"\meta_data.csv"))

    coins_prices = []
    coins_usd_volumes = []
    meta_data_df = meta_data_df[meta_data_df["ticker_2"] == "usd"]
    for _, row in meta_data_df.iterrows():
        filename = Path(DATA_FILES_PATH / row["filename"])
        coin_data_df = pd.read_csv(filename)
        coin_data_df["time"] = pd.to_datetime(coin_data_df["time"], unit="ms")
        coin_data_df = coin_data_df.set_index("time")
        coin_data_df["usd_volume"] = (
            coin_data_df["close"] * coin_data_df["volume"]
        )

        coins_prices.append(coin_data_df["close"].rename(row["ticker_1"]))
        coins_usd_volumes.append(
            coin_data_df["usd_volume"].rename(row["ticker_1"])
        )

    coins_prices_df = pd.concat(coins_prices, axis=1, join="outer")
    coins_prices_df = coins_prices_df.resample("1T").last()
    coins_usd_volumes_df = pd.concat(coins_usd_volumes, axis=1, join="outer")
    coins_usd_volumes_df = coins_usd_volumes_df.resample("1T").last()

    coins_prices_df.to_parquet(Path(WORKING_PATH + r"/coins_prices.parquet"))
    coins_usd_volumes_df.to_parquet(
        Path(WORKING_PATH + r"/coins_usd_volumes.parquet")
    )


if __name__ == "__main__":
    meta_data_df = generate_meta_data()
    meta_data_df.to_csv(Path(WORKING_PATH + r"/meta_data.csv"))

    generate_coins_data()
