'''
This module defines the class for the stock object.
'''

import yfinance as yf
import datetime as dt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go


class Stock:

    def __init__(self, ticker: str,
                 end_date: dt.date = dt.date.today()) -> None:
        '''
        Initializer function
        Downloads 1 year of stock data for Ticker given.
        Splits said data into training and test data, with testing data being
        the last 2 months.
        '''
        # Set Ticker String for later use
        self._ticker: str = ticker
        # Create dates for downloading and sorting
        self._year: int = 365
        self._cutoff_length: int = 60
        self._end_date: dt.date = end_date
        self._start_date: dt.date = self._end_date - \
            dt.timedelta(days=self._year)
        self._cutoff_date: dt.date = end_date - \
            dt.timedelta(days=self._cutoff_length)
        # Download stock price data using yf
        data = yf.download([ticker],
                           period='1y',
                           start=self._start_date,
                           end=self._end_date,
                           progress=False)
        # Remove unnecessary Data
        self._df: pd.DataFrame = data
        self._df = self._df.loc[:, self._df.columns != 'Close']
        # Train-Test Split
        self._train: pd.DataFrame = self._df[self._start_date:
                                             self._cutoff_date]
        self._test: pd.DataFrame = self._df[self._cutoff_date:self._end_date]
        self._train_f: pd.DataFrame =\
            self._train.loc[:, self._train.columns != 'Adj Close']
        self._train_l: pd.DataFrame = self._train['Adj Close']
        self._test_f: pd.DataFrame =\
            self._test.loc[:, self._test.columns != 'Adj Close']
        self._test_l: pd.DataFrame = self._test['Adj Close']
        self._future_train_labels = self._df['Adj Close']
        self._future_train_features =\
            self._df.loc[:, self._df.columns != 'Adj Close']

    def _run_knn(self,
                 train_f: pd.DataFrame,
                 test_f: pd.DataFrame,
                 train_l: pd.DataFrame,
                 test_l: pd.DataFrame) -> tuple[KNeighborsRegressor,
                                                npt.NDArray[np.float64],
                                                float]:
        '''
        Fits a KNN model given training and testing data pre-split.
        Returns a tuple containing:
        - The Model
        - The predicted data
        - THe Mean Squared Error
        '''
        # Create a K-Nearest Neighbors (KNN) model
        model = KNeighborsRegressor()
        model.fit(train_f, train_l)
        # Make predictions using the test data
        pred = model.predict(test_f)
        mse = mean_squared_error(pred, test_l)
        return (model, pred, mse)

    def _run_fr(self,
                train_f: pd.DataFrame,
                test_f: pd.DataFrame,
                train_l: pd.DataFrame,
                test_l: pd.DataFrame) -> tuple[RandomForestRegressor,
                                               npt.NDArray[np.float64],
                                               float]:
        '''
        Fits a RandomForest model given training and testing data pre-split.
        Returns a tuple containing:
        - The Model
        - The predicted data
        - THe Mean Squared Error
        '''
        # Create a Random Forest model
        model = RandomForestRegressor()
        model.fit(train_f, train_l)
        # Make predictions using the test data
        pred = model.predict(test_f)
        mse = mean_squared_error(pred, test_l)
        return (model, pred, mse)

    def run_models(self) -> None:
        '''
        Runs models and stores results as private fields for
        use with helper methods.
        Returns nothing.
        '''
        train_f_no_v = self._train_f.loc[:, self._train_f.columns != 'Volume']
        test_f_no_v = self._test_f.loc[:, self._test_f.columns != 'Volume']
        self._knn, self._knn_pred, self._knn_mse =\
            self._run_knn(train_f_no_v, test_f_no_v,
                          self._train_l, self._test_l)
        self._knn_v, self._knn_pred_v, self._knn_mse_v =\
            self._run_knn(self._train_f, self._test_f,
                          self._train_l, self._test_l)
        self._fr, self._fr_pred, self._fr_mse =\
            self._run_fr(train_f_no_v, test_f_no_v,
                         self._train_l, self._test_l)
        self._fr_v, self._fr_pred_v, self._fr_mse_v =\
            self._run_fr(self._train_f, self._test_f,
                         self._train_l, self._test_l)

    def get_data(self) -> dict[str, float]:
        '''
        Returns a dictionary containing the MSE values
        of the four tested models.
        '''
        out = {}
        out['KNN'] = self._knn_mse_v
        out['KNN_NO_VOLUME'] = self._knn_mse
        out['FR'] = self._fr_mse
        out['FR_NO_VOLUME'] = self._fr_mse_v
        return out

    def plot_predicted_vs_actual(self) -> None:
        '''
        Plots the predicted data versus
        the actual data. Returns nothing.
        '''
        try:
            layout = go.Layout(autosize=False, width=1500, height=500)
            models = ["knn", "knn_v", "fr", "fr_v"]
            model_names = ["KNN Predicted Prices (without Volume)",
                           "KNN Predicted Prices (with Volume)",
                           "Forest Random Predicted Prices (without Volume)",
                           "Forest Random Predicted Prices (with Volume)"]

            for model, model_name in zip(models, model_names):
                fig = go.Figure(layout=layout)
                fig.add_trace(go.Scatter(x=self._df.index, y=self._test_l,
                                         name='Actual Values',
                                         marker_color='blue'))

                if "knn" in model:
                    y_pred = self._knn_pred if "knn" \
                          in model else self._fr_pred
                else:
                    y_pred = self._knn_pred_v if "knn" \
                        in model else self._fr_pred_v

                fig.add_trace(go.Scatter(x=self._df.index, y=y_pred,
                                         name='Predicted Values',
                                         marker_color='red'))

                title = f"Actual {self._ticker} Stock Prices vs {model_name}"
                filename = f"plots/{self._ticker}_{model}.png"

                fig.update_layout(title=title, xaxis_title="Date",
                                  yaxis_title="Stock Price(USD)",
                                  legend_title="Legend")
                fig.write_image(filename)
        except Exception as e:
            print(f"Error occurred while plotting {model_name}: {str(e)}")

    def plot_future(self) -> None:
        '''
        Plots a future predicted data for a month in advance
        using the the Random Forest Regressor as it has the least
        amount of error.
        Returns nothing.
        '''
        try:
            self._future, self._future_pred, self._future_mse =\
                self._run_fr(self._future_train_features, self._test_f,
                             self._future_train_labels, self._test_l)\

            layout = go.Layout(autosize=False,
                               width=1500,
                               height=500)

            # plots knn without volume
            fig_future = go.Figure(layout=layout)
            fig_future.add_trace(go.Scatter(
                        x=[num for num in range(0, self._cutoff_length + 1)],
                        y=self._future_pred,
                        name='Future Values',
                        marker_color='blue'))
            fig_future.update_layout(title="Future " + self._ticker + " Stock\
    vs Predicted By Random Forest Regressor (with Volume)",
                                     xaxis_title="Days After Current Date",
                                     yaxis_title="Stock Price(USD)",
                                     legend_title="Legend")
            fig_future.write_image(f"plots/{self._ticker}_future.png")
        except Exception as e:
            print(f"Error occurred while plotting {self._ticker} future plot: \
                   {str(e)}")
