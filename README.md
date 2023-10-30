# Stock Market Prediction Using Machine Learning

## About

All-in-one stock price prediction models using scikit-learn.

Downloads data from Yahoo Finance, trains machine learning models using past data, outputs one month of predictions.

Built-in visualization methods using Plotly.

*CSE 163 Final Project (Winter 2023)*

## Setup

1. Install Dependences

```{python}
python3 -m pip install pandas numpy plotly yfinance scikit-learn pytest
```

2. Run test script to check

```{python}
python3 pytest
```

If the test script produces no output, then your environment is successfully configured!

## Usage

The Stock class located within `stock.py` contains all logic for downloading, running models and plotting results.

Example:

```{python}
from stock import Stock

aapl = Stock('AAPL')
aapl.run_models()  # Run ML Models
aapl.get_data()  # Return error rates
aapl.plot_predicted_vs_actual()  # Saves visualizations to 'plots' folder 
aapl.predict_future()  # Saves visualizations to 'plots' folder 
```

## Recreating Report Results

The file `stocks.txt` defines which stocks are included in the analysis.  Currently, they are the top 50 performing stocks.

To recreate the analysis used in our final report:
- Run `top50plots.py` to generate the visualizations for each stock.
- Run `analysis.py` to calculate which models were the most and least accurate.
    - Change the main function's parameter to `True` to recompute the results.csv file.  By default, the program uses the included file (the specific one we used in our analysis).

(The visualizations used in the report are predictions of AAPL price -- the example code in the usage section will generate them)

# INFO 443 PROJECT 1 REPORT

## Code Structure Analysis

### Project Summary

This project has three main areas where it aims to accurately train and predict stock market data. They are listed below.

Data Processing: Our preprocessing code will compile data based on the specified stock ticker. The module will parse the stock data from yfinance and process dates for each stock, meaning that our model will be run on a set of 365 days' worth of data. Running the models only on the most recent year is vital for relevance reasons. In terms of predicting the performance of a stock, which is one of our research questions, utilizing data for the stock back when it was worth less than a dollar when it could potentially be worth hundreds today wouldn’t be a great indicator of how it will perform in the future, so it is excluded. In order to compute the dates within Python efficiently, we utilized the library datetime. 

Machine Learning: This project primarily uses machine learning to predict stock market trends. Therefore, our selected model must meet the following requirements: understand the concept of Time, take in multiple days' worth of data, and output 30-60 data points representing future data. Our model's features will be the opening price, the high/low, and the volume. However, the volume will be one of our independent variables. We compared the inclusion of the volume in each model and how it affected accuracy. Our labels will be the adjusted closing prices. We will split our data to include 60 days' worth of data testing data, and the remaining 191 days will be the training data. This split will not be random because we are using a time-oriented dataset. Then the data will be separated into data with volume and without volume.

We will implement and compare two different machine learning models to see which model can more accurately predict stock market data. We chose our models after reviewing similar projects and landed on K-Nearest Neighbor Regression(KNN) and Random Forest Regression. The KNN works by finding the closest-looking data point to an input point and mapping them together. Meanwhile, the Random Forest Regressor works by aggregating all of the outputs of multiple decision trees and finding the average of the most optimal values. The models can then be run on our test data to see their accuracy. We will use mean squared error to test the accuracy. The most accurate model will then be used to predict future data. 

Visualization: To add a visual aid to our model, we will use the Plotly library to graph the predicted stock data versus the actual stock data. By plotting them on the same axis we are able to visualize the accuracy of our machine learning model. Then we will use the Plotly library to predict future data for 2 months in advance. We determined the Plotly library to be the best because it can plot data according to time in a much more aesthetic format that allows us to accurately see the date.


### Architectural Elements

The primary logic of the project lies within the `Stock` class. This class contains seven methods that are called on by other files within the project. The methods alongside their functionality is as follows:

`__init__(self, ticker: str, end_date: dt.date = dt.date.today()) -> None`
* Initializes a Stock object by downloading 1 year of stock data for a given ticker symbol. It splits the data into training and test datasets, with the test data representing the last 2 months of the data.

`_run_knn(self, train_f: pd.DataFrame, test_f: pd.DataFrame, train_l: pd.DataFrame, test_l: pd.DataFrame) -> tuple[KNeighborsRegressor, npt.NDArray[np.float64], float]`
* Fits a K-Nearest Neighbors (KNN) model given training and testing data pre-split. Returns a tuple containing the KNN model, the predicted data, and the Mean Squared Error (MSE).

`_run_fr(self, train_f: pd.DataFrame, test_f: pd.DataFrame, train_l: pd.DataFrame, test_l: pd.DataFrame) -> tuple[RandomForestRegressor, npt.NDArray[np.float64], float]`
* Fits a Random Forest (RF) model given training and testing data pre-split. Returns a tuple containing the Random Forest model, the predicted data, and the Mean Squared Error (MSE).

`run_models(self) -> None`
* Runs both KNN and Random Forest models using the training and test data and stores the results as private fields for later use with helper methods. It does this for both datasets with and without volume information.

`get_data(self) -> dict[str, float]`
* Returns a dictionary containing the Mean Squared Error (MSE) values of the four tested models, including KNN and Random Forest models with and without volume information.

`plot_predicted_vs_actual(self) -> None`
* Plots the predicted stock prices versus the actual stock prices for both KNN and Random Forest models, with and without volume information. It saves the plots as image files.

`plot_future(self) -> None`
* Plots the future predicted stock prices for a month in advance using the Random Forest Regressor, which is assumed to have the least amount of error. It saves the plot as an image file.

These functions have the capbility to clean up the data, run the machine learning models, and plot the data. They are then called by the aggregate_table process and the top_fifty_plots method. 

The aggregate table is a csv table created to look at the mean-squared error(mse) of each of the machine learning models. The mse is used to calculate the accuracy of the model. It is important to find the most accurate model so that we are able to accurately predict future data. This file primarily calls the `run_models` method from the stock class.

Meanwhile the top 50 plots file uses a list of the top 50 traded stocks on the United States Stock Market and plots five graphs for each stock. Four of these plots are the comparisions of the machine learning predicted data


![UML](images/Checkpoint_2_UML.jpg)
### Information Process FLows
![Data Flow](images/sequence_diagram.png)

## Architecure Analysis

Code Smells:

Long Method: The plot_predicted_vs_actual method is quite long and contains repetitive code for plotting different models. 

Hardcoded Values / Primitve Obsession: There are some hardcoded values, such as 60 in fig_future.add_trace, which could be replaced with named constants for better readability and maintainability.

No Error Handling in Plotting: The code for plotting doesn't handle potential errors when saving plot images. It's important to handle exceptions that might occur during file operations.

Unecessary Use of  Global Main Function: Analysis is done within the main function, this code can be rewritten as it's own function to be called in main. 

Lack of Comments/Documentation: While there are some docstrings present, there are still some parts of the code that could benefit from more comments or explanations. For instance, the purpose of the private fields and their naming conventions could be clarified.

Inconsistent Naming: cutoff_d meanwhile others have date.



## Automated Tests

## Refactoring Code

## INFO 443 Checkpoint 2

Here are the UML and Data Flow diagrams for checkpoint 2.
![UML](images/Checkpoint_2_UML.jpg)
![Data Flow](images/sequence_diagram.png)

## INFO 443 Checkpoint 3

To test this code, first open the test_stock.py file and simply type pytest in the command line. Thank you!

## Acknowledgements

- [yfinance](https://pypi.org/project/yfinance/), used to download price data from Yahoo finance
- Other packages used: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/stable/), [Plotly Python Library](https://plotly.com/python/)