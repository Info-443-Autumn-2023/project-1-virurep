"""
This module tests the functionality of the `Stock` class defined in stock.py.
It accesses private fields which should not be used in any end-user
program.

To test functionality, this program will create `Stock` instance using the
ticker 'F' (Ford), then check and see if any errors occur during execution
of various methods.  The program will crash if any errors occur.

This file does *not* directly test the logic of scikit-learn's ML model
implementations, instead it checks if your environment is correctly
set up for the use of the `Stock` class.

Utilizes the assert_equals function from the CSE 163 assignments environment.
(Downloaded from A4)
"""

from stock import Stock
import datetime as dt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import float64
import os
from aggregate_table import agg_table
import pandas as pd
from analysis import write_agg_csv
import pytest


def test_initializer() -> None:
    """
    Tests the initializer method of the `Stock` class.
    Returns none, crashes if any tests fail
    """
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    assert f_s._ticker == "F", "Initializer Error - Ticker Match Incorrect"
    assert f_s._end_date == dt.date(
        2022, 10, 5
    ), "Initializer Error - Date \
          Error"
    assert f_s._start_date == dt.date(
        2021, 10, 5
    ), "Initializer Error - Date \
          Error"
    assert f_s._cutoff_d == dt.date(
        2022, 8, 6
    ), "Initializer Error - Date \
          Error"
    assert (
        len(f_s._df.columns) == 5
    ), "Initializer Error - Data Frame Read \
          Error"
    assert (
        len(f_s._train) + len(f_s._test)
    ) == 252, "Initializer Error-Data\
          Split Error"
    print("All Initializer Tests Passed")


def test_knn():
    """
    Tests the _run_KNN method of the `Stock` class (Implementation of the
    KNN model)
    Returns none, crashes if any tests fail.
    """
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    out = f_s._run_knn(f_s._train_f, f_s._test_f, f_s._train_l, f_s._test_l)
    assert type(out[0]) is type(KNeighborsRegressor())
    assert len(out[1]) == 41
    assert type(out[2]) is float64


def test_fr():
    """
    Tests the _run_fr method of the `Stock` class (Implementation of the
    Forest Regressor model)
    Returns none, crashes if any tests fail.
    """
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    out = f_s._run_fr(f_s._train_f, f_s._test_f, f_s._train_l, f_s._test_l)
    assert type(out[0]) is type(RandomForestRegressor())
    assert len(out[1]) == 41
    assert type(out[2]) is float64


def test_output_data() -> None:
    """
    Tests the data returned from the `Stock` class after all models are
    computed.
    Returns none, crashes if any tests fail.
    """
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    f_s.run_models()
    data = f_s.get_data()
    exp_keys = ["KNN", "KNN_NO_VOLUME", "FR", "FR_NO_VOLUME"]
    assert list(data.keys()) == exp_keys
    assert len(data) == 4
    assert type(data["KNN"]) is float64
    assert type(data["KNN_NO_VOLUME"]) is float64
    assert type(data["FR"]) is float64
    assert type(data["FR_NO_VOLUME"]) is float64


def test_plot_predicted_vs_actual():
    '''
    Tests the plotting method that plots the predicted vs actual values
    for the `Stock` class. Returns none, crashes if any tests fail.
    '''
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    f_s.run_models()
    f_s.plot_predicted_vs_actual()
    ticker = f_s._ticker
    plot_files = [
        f"plots/{ticker}_knn.png",
        f"plots/{ticker}_knn_v.png",
        f"plots/{ticker}_fr.png",
        f"plots/{ticker}_fr_v.png",
    ]

    for plot_file in plot_files:
        assert os.path.isfile(plot_file), f"Plot file {plot_file} not found"


def test_plot_future():
    '''
    Tests the plotting method that plots the future values
    for the `Stock` class. Returns none, crashes if any tests fail.
    '''
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    f_s.run_models()
    f_s.plot_future()

    assert hasattr(f_s, "_future")
    assert hasattr(f_s, "_future_pred")
    assert hasattr(f_s, "_future_mse")

    ticker = f_s._ticker
    plot_file = f"plots/{ticker}_future.png"
    assert os.path.isfile(plot_file), f"Plot file {plot_file} not found"


SAMPLE_FILENAME = "sample_stocks.txt"


def test_agg_table_generation():
    '''
    Tests the aggregate table function to see if the table is
    generated properly, crashes if tests fail.
    '''

    with open(SAMPLE_FILENAME, "w") as f:
        f.write("AAPL\nMSFT\nGOOGL\n")
    aggregate_table = agg_table(SAMPLE_FILENAME)
    print(aggregate_table.index)
    expected_tickers = ["AAPL", "MSFT", "GOOGL"]
    expected_model_names = ["KNN", "KNN_NO_VOLUME", "FR", "FR_NO_VOLUME"]

    assert isinstance(aggregate_table, pd.DataFrame)
    assert aggregate_table.index.to_list() == expected_tickers
    assert all(column in aggregate_table.columns for column in
               expected_model_names)
    assert aggregate_table.dtypes.eq(float).all()

    # Clean up the sample input file
    os.remove(SAMPLE_FILENAME)


@pytest.fixture(scope="function")
def prepare_test_env():
    # Set up (before test)
    test_csv_file = 'test_results.csv'
    yield test_csv_file  # Return the file name for the testpyte


# Test the write_agg_csv function
def test_write_agg_csv(prepare_test_env):
    # Run the function
    write_agg_csv()

    # Check if the CSV file was created
    assert os.path.exists('results.csv')


# # Test the main function
# def test_main(prepare_test_env):
#     # Prepare a test CSV file for the 'main' function

#     # Run the main function with 'data' set to True
#     main(data=True)

#     df = pd.read_csv(test_csv_file)
#     assert 'Best Model' in df.columns
#     assert 'Worst Model' in df.columns
