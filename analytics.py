import functools
import pickle as pkl
from pathlib import Path
from typing import Tuple, Sequence

import pandas as pd
import numpy as np
import matplotlib.dates as mpldt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator


WORKING_PATH = r"C:\Dennis\crypto"


def inspect_strategy_output(output_path: Path, strategy_name: str):
    with open(output_path, "rb") as f:
        (
            target_pos_df,
            actual_pos_df,
            portfolio_pnls_df,
            portfolio_summary_df,
        ) = pkl.load(f)

    strategy_gross_pnl = portfolio_summary_df["gross_pnl"].resample("D").sum()
    strategy_net_pnl = portfolio_summary_df["net_pnl"].resample("D").sum()
    strategy_gmv = portfolio_summary_df["gmv"].resample("D").sum()

    strategy_metrics = _calc_metrics(
        strategy_gross_pnl,
        strategy_net_pnl,
        strategy_gmv,
    )
    pnl_plot = plot_pnls(strategy_metrics, strategy_name)
    rolling_sharpes = compute_rolling_sharpes(strategy_net_pnl, [1, 3, 6, 12])
    rolling_shapres_plot = plot_rolling_sharpes(rolling_sharpes, strategy_name)

    pnl_plot.savefig(Path(WORKING_PATH + f"/plots/{strategy_name}_pnl.png"))
    rolling_shapres_plot.savefig(
        Path(WORKING_PATH + f"/plots/{strategy_name}_rolling_sharpes.png")
    )


def combine_strategies(
    strategy_names: Sequence[str], combined_strategy_name: str
):
    gross_pnls = []
    net_pnls = []
    gmvs = []

    for substrategy_name in strategy_names:
        strategy_output_path = Path(
            WORKING_PATH + f"/output/{substrategy_name}.pkl"
        )
        with open(strategy_output_path, "rb") as f:
            (
                _,
                _,
                _,
                portfolio_summary_df,
            ) = pkl.load(f)

        gross_pnls.append(
            portfolio_summary_df["gross_pnl"].resample("D").sum()
        )
        net_pnls.append(portfolio_summary_df["net_pnl"].resample("D").sum())
        gmvs.append(portfolio_summary_df["gmv"].resample("D").sum())

    combined_gross_pnls = functools.reduce(
        lambda df1, df2: df1.add(df2, fill_value=0), gross_pnls
    )
    combined_net_pnls = functools.reduce(
        lambda df1, df2: df1.add(df2, fill_value=0), net_pnls
    )
    combine_gmvs = functools.reduce(
        lambda df1, df2: df1.add(df2, fill_value=0), gmvs
    )

    combined_strategy_metrics = _calc_metrics(
        combined_gross_pnls,
        combined_net_pnls,
        combine_gmvs,
    )

    pnl_plot = plot_pnls(combined_strategy_metrics, combined_strategy_name)
    rolling_sharpes = compute_rolling_sharpes(combined_net_pnls, [1, 3, 6, 12])
    rolling_shapres_plot = plot_rolling_sharpes(
        rolling_sharpes, combined_strategy_name
    )

    pnl_plot.savefig(
        Path(WORKING_PATH + f"/plots/{combined_strategy_name}_pnl.png")
    )
    rolling_shapres_plot.savefig(
        Path(
            WORKING_PATH
            + f"/plots/{combined_strategy_name}_rolling_sharpes.png"
        )
    )


# Utils
def set_annual_xticks(
    ax: Axes,
    minor: bool = False,
    rotation: float = None,
) -> None:
    """Set the x-axis ticks to be annual"""
    xaxis = ax.xaxis
    xaxis.set_major_locator(YearLocator())
    xaxis.set_major_formatter(DateFormatter("%y"))

    if minor:
        xaxis.set_minor_locator(MonthLocator())

    if rotation is not None:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation(rotation)

        fig = ax.get_figure()
        fig.subplots_adjust(bottom=0.2)


def plot_pnls(metrics, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title(title)

    series = [
        metrics["cum_pl_gross"],
        metrics["cum_pl_net"],
    ]

    for s in series:
        ax.plot(
            mpldt.date2num(s.index),
            s.values,
            label=s.name,
            lw=1,
        )

    ax.grid(True)
    ax.axhline(0, color="gainsboro")
    lines, labels = ax.get_legend_handles_labels()
    lines = [line for i, line in enumerate(lines)]
    labels = [label for i, label in enumerate(labels)]
    ax.legend(lines, labels, loc=0)

    set_annual_xticks(ax)

    stats_f_text = (
        "Sharpe: {:.2f} ({:.2f})\n"
        "Avg GMV: {:.2f}\n"
        "Avg Ret on GMV: {:.1%} ({:.1%})\n"
    )
    ax.text(
        0.60,
        0.07,
        stats_f_text.format(
            metrics["sharpe"] * np.sqrt(360),
            metrics["sharpe_net"] * np.sqrt(360),
            metrics["avg_gmv"],
            metrics["pl_per_gross"] * 360,
            metrics["pl_net_per_gross"] * 360,
        ),
        bbox={"facecolor": "white", "pad": 5, "alpha": 0.8},
        transform=ax.transAxes,
    )
    return fig


def plot_rolling_sharpes(
    rolling_sharpes: Sequence[pd.Series], strategy_name: str
):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title(f"{strategy_name} Rolling Sharpes")

    for s in rolling_sharpes:
        ax.plot(
            mpldt.date2num(s.index),
            s.values,
            label=s.name,
            lw=1,
        )

    ax.grid(True)
    ax.axhline(0, color="gainsboro")
    lines, labels = ax.get_legend_handles_labels()
    lines = [line for i, line in enumerate(lines)]
    labels = [label for i, label in enumerate(labels)]
    ax.legend(lines, labels)

    set_annual_xticks(ax)

    return fig


def compute_rolling_sharpes(
    net_pnls,
    windows_list: Sequence[int],  # in Months
):
    annualizer = 360
    month_multiplier = 30

    rolling_sharpes_series = []
    for window in windows_list:
        window_length = int(np.round(month_multiplier * window))
        sharpe = (
            net_pnls.rolling(
                window_length, min_periods=int(0.7 * window_length)
            ).mean()
            / net_pnls.rolling(
                window_length, min_periods=int(0.7 * window_length)
            ).std()
            * np.sqrt(annualizer)
        )
        rolling_sharpes_series.append(sharpe.rename(f"{window}-Month"))

    return rolling_sharpes_series


def _calc_drawdowns(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = series.index
    maxes = np.fmax.accumulate(series)
    maxes_list = list(maxes)
    drawdowns = maxes - series

    idx_list = list(idx)
    max_idx = [idx_list[maxes_list.index(m)] for m in maxes]
    drawstarts = pd.Series(data=max_idx, index=idx)
    return drawdowns, drawstarts


def _calc_metrics(gross_pnls, net_pnls, gmvs):
    avg_gmv = gmvs.mean()
    max_gmv = gmvs.max()
    pos_sum = gmvs.sum()

    cum_gross_pnls = gross_pnls.cumsum()
    cum_net_pnls = net_pnls.cumsum()

    if pos_sum:
        total_gross_pl = gross_pnls.sum()
        pl_per_gross = total_gross_pl / pos_sum

        non_zero_pl = gross_pnls[gross_pnls != 0]
        vol = non_zero_pl.std()
        sharpe = non_zero_pl.mean() / vol

        total_pl_net = net_pnls.sum()
        pl_net_per_gross = total_pl_net / pos_sum

        non_zero_pl_net = net_pnls[net_pnls != 0]
        vol_net = non_zero_pl_net.std()
        sharpe_net = non_zero_pl_net.mean() / vol_net

        drawdowns, drawstarts = _calc_drawdowns(cum_net_pnls)
        maxdraw = np.max(drawdowns)
        drawend = (drawdowns == maxdraw).idxmax()
        drawstart = drawstarts.loc[drawend]
        drawdown = pd.Series(
            data=[maxdraw, drawstart, drawend],
            index=["Max Draw", "Start", "End"],
        )
        drawstart_gmvs = pd.Series(
            gmvs.reindex(drawstarts.values).values, drawstarts.index
        )
        pct_drawdowns = np.divide(drawdowns, drawstart_gmvs)

    else:
        pl_per_gross = np.nan
        pl_net_per_gross = np.nan
        vol = np.nan
        vol_net = np.nan
        sharpe = np.nan
        sharpe_net = np.nan
        drawdown = np.nan
        drawdowns = np.nan
        pct_drawdowns = np.nan

    return {
        "cum_pl_gross": cum_gross_pnls,
        "cum_pl_net": cum_net_pnls,
        "gmv": gmvs,
        "vol": vol,
        "vol_net": vol_net,
        "sharpe": sharpe,
        "sharpe_net": sharpe_net,
        "avg_gmv": avg_gmv,
        "max_gmv": max_gmv,
        "pl_per_gross": pl_per_gross,
        "pl_net_per_gross": pl_net_per_gross,
        "drawdown": drawdown,
        "dollar_drawdown_series": drawdowns,
        "pct_drawdown_series": pct_drawdowns,
    }


if __name__ == "__main__":
    # strategy_name = "daily_momentum"
    # strategy_output_path = Path(WORKING_PATH + f"/output/{strategy_name}.pkl")
    # inspect_strategy_output(strategy_output_path, strategy_name)

    good_strategies = [
        "30_minutes_momentum",
        "1_hours_momentum",
        "2_hours_momentum",
        "4_hours_momentum",
        "8_hours_mean_reversion",
    ]
    combine_strategies(good_strategies, "combined_intraday_strategy")
