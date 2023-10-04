'''
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
'''

from stock import Stock
from cse163_utils import assert_equals

import datetime as dt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import float64


def test_initializer(f_s: Stock) -> None:
    '''
    Tests the initializer method of the `Stock` class.
    Returns none, crashes if any tests fail
    '''
    assert_equals(f_s._ticker, 'F')
    assert_equals(f_s._end_date, dt.date(2022, 10, 5))
    assert_equals(f_s._start_date, dt.date(2021, 10, 5))
    assert_equals(f_s._cutoff_d, dt.date(2022, 8, 6))
    assert_equals(len(f_s._df.columns), 5)
    assert_equals((len(f_s._train) + len(f_s._test)), 252)


def test_knn(f_s: Stock) -> None:
    '''
    Tests the _run_KNN method of the `Stock` class (Implementation of the
    KNN model)
    Returns none, crashes if any tests fail.
    '''
    out = f_s._run_knn(f_s._train_f, f_s._test_f, f_s._train_l, f_s._test_l)
    assert_equals(type(out[0]), type(KNeighborsRegressor()))
    assert_equals(len(out[1]), 41)
    assert_equals(type(out[2]), float64)


def test_fr(f_s: Stock) -> None:
    '''
    Tests the _run_fr method of the `Stock` class (Implementation of the
    FR model)
    Returns none, crashes if any tests fail.
    '''
    out = f_s._run_fr(f_s._train_f, f_s._test_f, f_s._train_l, f_s._test_l)
    assert_equals(type(out[0]), type(RandomForestRegressor()))
    assert_equals(len(out[1]), 41)
    assert_equals(type(out[2]), float64)


def test_output_data(f_s: Stock) -> None:
    '''
    Tests the data returned from the `Stock` class after all models are
    computed.
    Returns none, crashes if any tests fail.
    '''
    data = f_s.get_data()
    exp_keys = ['KNN', 'KNN_NO_VOLUME', 'FR', 'FR_NO_VOLUME']
    assert_equals(list(data.keys()), exp_keys)
    assert_equals(len(data), 4)
    assert_equals(type(data['KNN']), float64)
    assert_equals(type(data['KNN_NO_VOLUME']), float64)
    assert_equals(type(data['FR']), float64)
    assert_equals(type(data['FR_NO_VOLUME']), float64)


def main(f_s: Stock) -> None:
    '''
    Main function that runs all tests.
    Returns none.
    '''
    test_initializer(f_s)
    test_knn(f_s)
    test_fr(f_s)
    f_s.run_models()
    test_output_data(f_s)


if __name__ == '__main__':
    FORD = Stock('F', end_date=dt.date(2022, 10, 5))
    main(FORD)