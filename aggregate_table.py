'''
This module compiles an aggregate table of each stock's MSE for each
Machine Learning model that was created in the "stock" module.
'''
# Import Statements
from stock import Stock
import datetime as dt
import pandas as pd


def agg_table(filename: str):
    """
    Takes a text document filename that contains the names of stocks that are
    to be analyzed

    Returns an aggregate table that contains each stock's mean squared error
    for KNN, KNN (minus volume factor), Forest Regression, & Forest Regression
    (minus volume factor).
    """
    a_table = []

    # Constructs list of reformatted stock ticker names
    with open(filename) as f:
        stocks_list = [stock.strip('\n') for stock in f.readlines()]

    # Runs all ML models (from stock file) on current ticker
    for ticker in stocks_list:
        current_t = Stock(ticker, end_date=dt.date(2023, 3, 6))
        current_t.run_models()
        a_table.append(current_t.get_data())  # Adds to cumulative list

    # Converts list of dicts to pd df with cols as ML models & indx as Tickers
    a_table = pd.DataFrame.from_records(a_table, index=stocks_list)

    return a_table


def main() -> None:
    agg_table('stocks.txt')

    return None


if __name__ == '__main__':
    main()
