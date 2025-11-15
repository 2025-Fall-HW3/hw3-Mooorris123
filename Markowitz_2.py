"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0, momentum_lookback=126, volatility_lookback=50, num_top_assets=5, market_trend_lookback=200):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.num_top_assets = num_top_assets
        self.market_trend_lookback = market_trend_lookback

        self.max_lookback = max(self.momentum_lookback, self.volatility_lookback, self.market_trend_lookback)

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        spy_sma = self.price['SPY'].rolling(window=self.market_trend_lookback).mean()

        for i in range(self.max_lookback + 1, len(self.price)):
            current_date = self.price.index[i]

            is_bull_market = self.price['SPY'].iloc[i] > spy_sma.iloc[i]

            if is_bull_market:
                momentum_window = self.returns[assets].iloc[i - self.momentum_lookback : i]
                momentum_scores = (1 + momentum_window).prod() - 1

                top_assets = momentum_scores.nlargest(self.num_top_assets).index.tolist()

                volatility_window = self.returns[top_assets].iloc[i - self.volatility_lookback : i]
                volatility = volatility_window.std()
                volatility.replace(0, 1e-10, inplace=True)

                inverse_volatility = 1 / volatility
                final_weights = inverse_volatility / inverse_volatility.sum()

                self.portfolio_weights.loc[current_date, assets] = 0
                self.portfolio_weights.loc[current_date, top_assets] = final_weights.values

            else:
                self.portfolio_weights.loc[current_date, assets] = 0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    # All grading logic is protected in grader_2.py
    judge.run_grading(args)