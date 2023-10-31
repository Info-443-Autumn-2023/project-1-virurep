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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import float64
from aggregate_table import agg_table
from analysis import write_agg_csv
from top50plots import stocks_processing
import pytest
import pandas as pd
import os
import datetime as dt


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
    Tests the _run_KNN method of the `Stock` class.
    Ensures that the method returns the expected results and handles errors.
    """
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    out = f_s._run_knn(f_s._train_f, f_s._test_f, f_s._train_l, f_s._test_l)

    assert isinstance(out[0], KNeighborsRegressor), "Expected a\
          KNeighborsRegressor instance."

    expected_length = 41
    assert len(out[1]) == expected_length, f"Expected a length of\
          {expected_length}."

    assert isinstance(out[2], float), "Expected a float value."


def test_fr():
    """
    Tests the _run_fr method of the `Stock` class (Implementation of the
    Forest Regressor model)
    Returns none, crashes if any tests fail.
    """
    f_s = Stock("F", end_date=dt.date(2022, 10, 5))
    out = f_s._run_fr(f_s._train_f, f_s._test_f, f_s._train_l, f_s._test_l)

    assert type(out[0]) is type(RandomForestRegressor())

    expected_length = 41
    assert len(out[1]) == expected_length, f"Expected a length of \
          {expected_length} but got {len(out[1])}."

    assert type(out[2]) is float64, f"Expected a \
          {float64} but got {type(out[2])}."


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
    assert list(data.keys()) == exp_keys, f"Expected keys: {exp_keys},\
          but got keys: {list(data.keys())}"

    expected_length = 4
    assert len(data) == expected_length, f"Expected a length of\
          {expected_length}, but got a length of {len(data)}."

    assert isinstance(data["KNN"], float), "Expected a float type for 'KNN'\
          value."
    assert isinstance(data["KNN_NO_VOLUME"], float), "Expected a float type\
          for 'KNN_NO_VOLUME' value."
    assert isinstance(data["FR"], float), "Expected a float type for 'FR'\
          value."
    assert isinstance(data["FR_NO_VOLUME"], float), "Expected a float type for\
          'FR_NO_VOLUME' value."


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

    assert isinstance(aggregate_table, pd.DataFrame), \
           "Expected 'aggregate_table' to be a pandas DataFrame."
    assert aggregate_table.index.to_list() == expected_tickers, "Table Index\
        Incorrect"
    assert all(column in aggregate_table.columns for column in
               expected_model_names), "Aggregate Table Error"
    assert aggregate_table.dtypes.eq(float).all(), "Aggregate Table Type Error"

    # Clean up the sample input file
    os.remove(SAMPLE_FILENAME)


# Define test data and filenames for agg_csv test
test_input_file = "test_stocks.txt"
test_output_file = "test_results.csv"


@pytest.fixture
def sample_data():
    '''
    Creates Sample Data for testing.
    '''
    data = {
        "Stock": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "KNN": [0.95, 0.92, 0.91, 0.93],
        "KNN_NO_VOLUME": [0.94, 0.91, 0.90, 0.92],
        "FR": [0.97, 0.93, 0.92, 0.95],
        "FR_NO_VOLUME": [0.96, 0.92, 0.91, 0.94]
    }
    df = pd.DataFrame(data)
    return df


def test_write_agg_csv(capsys, sample_data):
    '''
    Test the write_agg_csv function.
    This test verifies that the write_agg_csv function correctly
    generates an output CSV file, checks the data type of the returned
    DataFrame, captures the printed output, and checks for the presence
    of expected output messages and DataFrame columns.
    '''
    sample_data.to_csv(test_input_file, index=False)
    df = write_agg_csv()
    assert os.path.exists('results.csv')
    assert isinstance(df, pd.DataFrame), "Function did not return a\
          DataFrame"
    captured = capsys.readouterr()
    # Capture the printed output and check if it contains expected results
    assert "Number of times each model was the most accurate" in \
           captured.out,  "Incorrect Output"
    assert "Number of times each model was the least accurate" in \
           captured.out,  "Incorrect Output"
    assert 'Best Model' in df.columns, "Dataframe Error"
    assert 'Worst Model' in df.columns, "Dataframe Error"


def test_stocks_processing():
    '''
    Test the stocks_processing function.
    This test verifies that the stocks_processing function correctly processes
    a sample stocks file, creates output files in the "plots" folder, and
    checks if the expected files are present in the "plots" folder.
    '''
    sample_file = "sample_stocks.txt"
    with open(sample_file, "w") as f:
        f.write("AAPL\nMSFT\nGOOGL\n")
    stocks_processing(sample_file)

    plots_folder = "plots"
    files_in_plots = os.listdir(plots_folder)
    expected_files = ["AAPL_knn.png",
                      "GOOG_knn.png",
                      "MSFT_knn.png"]

    for expected_file in expected_files:
        assert expected_file in files_in_plots, f"'{expected_file}' is\
              not in the 'plots' folder."

    os.remove(test_input_file)
