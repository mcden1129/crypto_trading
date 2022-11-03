import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd


WORKING_PATH = r"C:\Dennis\crypto"


def cross_sectional_strategy(resample_freq: str, name: str, momentum: bool):
    (
        coins_universe,
        lookback_window,
        px_ffill_limit_in_minute,
        avg_usd_volume_thres,
        has_traded_cnt_thres,
        executables_cnt_thres,
        trade_pct_of_volume,
        long_short_percentile,
        simple_cost_in_bps,
    ) = get_config(resample_freq)
    usd_trade_size_per_side = (avg_usd_volume_thres * trade_pct_of_volume) * (
        executables_cnt_thres * long_short_percentile
    )

    # Load Data
    coins_prices_df = pd.read_parquet(
        Path(WORKING_PATH + r"/coins_prices.parquet")
    )
    coins_usd_volumes_df = pd.read_parquet(
        Path(WORKING_PATH + r"/coins_usd_volumes.parquet")
    )

    # Resample Data
    coins_prices_df = (
        coins_prices_df[coins_universe]
        .ffill(limit=px_ffill_limit_in_minute)
        .resample(resample_freq)
        .last()
    )
    coins_usd_volumes_df = (
        coins_usd_volumes_df[coins_universe].resample(resample_freq).sum()
    )

    coins_ret_df = np.log(coins_prices_df / coins_prices_df.shift(1))

    # Executability Check
    rolling_has_traded_cnt = coins_usd_volumes_df.rolling(
        lookback_window
    ).count()
    rolling_avg_usd_volume = (
        coins_usd_volumes_df.fillna(0).rolling(lookback_window).mean()
    )
    valid_ret = coins_ret_df.notnull()

    executables = (
        (rolling_has_traded_cnt > has_traded_cnt_thres)
        & (rolling_avg_usd_volume > avg_usd_volume_thres)
        & valid_ret
    )
    executables_cnt = executables.sum(axis=1)

    executables_period = executables_cnt.where(
        executables_cnt >= executables_cnt_thres
    ).dropna()
    strategy_start = executables_period.index.min()
    strategy_end = executables_period.index.max()

    executables_coins_ret_df = coins_ret_df[strategy_start:strategy_end].where(
        executables
    )
    executables_coins_rank_df = executables_coins_ret_df.rank(
        axis=1, method="min", na_option="keep"
    )

    # Use Rank of Prev Period to Guide Curr Period Position
    executables_coins_prev_rank_df = executables_coins_rank_df.shift(1)

    target_pos = {}
    actual_pos = {}
    portfolio_info = {}
    all_coins = list(executables_coins_rank_df.columns)
    for t, row in executables_coins_prev_rank_df.iterrows():
        if t in executables_period.index:
            cnt = executables_cnt.loc[t]
            num_coins = int(cnt * 0.10)
            rank_sorted = row.dropna().sort_values()

            valid_coins = list(row.dropna().index)
            if momentum:
                # Momentum - Buy Past Winners, Short Past Losers
                buy_coins = list(rank_sorted[0:num_coins].index)
                sell_coins = list(rank_sorted[-num_coins:].index)
            else:
                # Mean Reversion - Buy Past Losers, Short Past Winners
                buy_coins = list(rank_sorted[-num_coins:].index)
                sell_coins = list(rank_sorted[0:num_coins].index)

            target = [0] * executables_coins_rank_df.shape[1]
            actual = [0] * executables_coins_rank_df.shape[1]
            for buy in buy_coins:
                target[all_coins.index(buy)] = usd_trade_size_per_side / len(
                    buy_coins
                )
                actual[all_coins.index(buy)] = min(
                    coins_usd_volumes_df.shift(-1).loc[t, buy] * 0.15,
                    usd_trade_size_per_side / len(buy_coins),
                )
            for sell in sell_coins:
                target[all_coins.index(sell)] = -usd_trade_size_per_side / len(
                    sell_coins
                )
                actual[all_coins.index(sell)] = -min(
                    coins_usd_volumes_df.shift(-1).loc[t, sell] * 0.15,
                    usd_trade_size_per_side / len(sell_coins),
                )

            portfolio_info[t] = (valid_coins, buy_coins, sell_coins)

            target_pos[t] = target
            actual_pos[t] = actual

    target_pos_df = pd.DataFrame.from_dict(
        target_pos, orient="index", columns=all_coins
    )
    actual_pos_df = pd.DataFrame.from_dict(
        actual_pos, orient="index", columns=all_coins
    )
    portfolio_pnls_df = (
        actual_pos_df.shift(1) * coins_ret_df.loc[actual_pos_df.index]
    )
    portfolio_summary_df = pd.DataFrame.from_dict(
        portfolio_info, orient="index", columns=["valid_coins", "buy", "sell"]
    )
    portfolio_summary_df["lmv"] = actual_pos_df.where(actual_pos_df > 0).sum(
        axis=1
    )
    portfolio_summary_df["smv"] = actual_pos_df.where(actual_pos_df < 0).sum(
        axis=1
    )
    portfolio_summary_df["gmv"] = actual_pos_df.abs().sum(axis=1)

    portfolio_summary_df["gross_pnl"] = portfolio_pnls_df.sum(axis=1)
    portfolio_summary_df["cum_gross_pnl"] = portfolio_summary_df[
        "gross_pnl"
    ].cumsum()

    portfolio_summary_df["net_pnl"] = portfolio_summary_df[
        "gross_pnl"
    ] - portfolio_summary_df["gmv"] * (simple_cost_in_bps / 10000)
    portfolio_summary_df["cum_net_pnl"] = portfolio_summary_df[
        "net_pnl"
    ].cumsum()

    strategy_output = (
        target_pos_df,
        actual_pos_df,
        portfolio_pnls_df,
        portfolio_summary_df,
    )
    with open(Path(WORKING_PATH + f"/output/{name}.pkl"), "wb") as f:
        pkl.dump(strategy_output, f)


def get_config(resample_freq: str):
    lookback_window = 30
    px_ffill_limit_in_minute = 15

    time_series_length_thres_1_min = 60 * 24 * 60
    meta_data = pd.read_csv(
        Path(WORKING_PATH + r"/meta_data.csv")
    )  # 1-Minute Meta Data
    meta_data_filter = (meta_data["ticker_2"] == "usd") & (
        meta_data["n_row"] >= time_series_length_thres_1_min
    )
    coins_universe = list(meta_data[meta_data_filter]["ticker_1"])

    has_traded_cnt_thres = int(0.50 * lookback_window)
    executables_cnt_thres = 30

    avg_daily_usd_volume_thres = 10000
    if resample_freq == "1T":
        avg_usd_volume_thres = avg_daily_usd_volume_thres / 24 / 60
    elif resample_freq == "5T":
        avg_usd_volume_thres = 5 * avg_daily_usd_volume_thres / 24 / 60
    elif resample_freq == "15T":
        avg_usd_volume_thres = 15 * avg_daily_usd_volume_thres / 24 / 60
    elif resample_freq == "30T":
        avg_usd_volume_thres = 30 * avg_daily_usd_volume_thres / 24 / 60
    elif resample_freq == "1h":
        avg_usd_volume_thres = avg_daily_usd_volume_thres / 24
    elif resample_freq == "2h":
        avg_usd_volume_thres = 2 * avg_daily_usd_volume_thres / 24
    elif resample_freq == "4h":
        avg_usd_volume_thres = 4 * avg_daily_usd_volume_thres / 24
    elif resample_freq == "6h":
        avg_usd_volume_thres = 6 * avg_daily_usd_volume_thres / 24
    elif resample_freq == "8h":
        avg_usd_volume_thres = 8 * avg_daily_usd_volume_thres / 24
    elif resample_freq == "D":
        avg_usd_volume_thres = avg_daily_usd_volume_thres
    else:
        raise ValueError(
            f"Frequency <{resample_freq}> is not valid/ supported."
        )
    trade_pct_of_volume = 0.15
    long_short_percentile = 0.10

    simple_cost_in_bps = 3  # Transaction Cost/ Slippage in bps

    return (
        coins_universe,
        lookback_window,
        px_ffill_limit_in_minute,
        avg_usd_volume_thres,
        has_traded_cnt_thres,
        executables_cnt_thres,
        trade_pct_of_volume,
        long_short_percentile,
        simple_cost_in_bps,
    )


if __name__ == "__main__":
    resample_freq = "30T"

    cross_sectional_strategy(
        resample_freq, name="30_minutes_momentum", momentum=True
    )
    cross_sectional_strategy(
        resample_freq, name="30_minutes_mean_reversion", momentum=False
    )
