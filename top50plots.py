'''
Runs the stock plots on the Top 50 most
publicly traded companies. Saves the plots
to the plots folder.
'''

from stock import Stock
import datetime as dt


def stocks_processing(filename: str) -> None:
    '''
    Opens the stock file and iterates through the
    tickers. Creates a stock object for each ticker,
    and runs the machine learning models on them. Then plots
    all 5 graphs for each ticker
    '''
    with open(filename) as f:
        stocks_list = [stock.strip('\n') for stock in f.readlines()]

    for ticker in stocks_list:
        analysis_stock = Stock(ticker=ticker,
                               end_date=dt.date(2022, 12, 15))
        analysis_stock.run_models()
        analysis_stock.plot_predicted_vs_actual()
        analysis_stock.plot_future()


def main() -> None:
    stocks_file = 'stocks.txt'
    stocks_processing(stocks_file)


if __name__ == "__main__":
    main()
