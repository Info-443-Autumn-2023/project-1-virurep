'''
This file contains code for analyzing the results of our Machine learning
models.  The statistics calculated in this file were used in the 'Results'
section of the Report PDF.
'''

from aggregate_table import agg_table
import pandas as pd


def write_agg_csv() -> None:
    '''
    Writes aggregate CSV file using agg_table
    '''
    df = agg_table('stocks.txt')
    df.to_csv('results.csv', encoding='utf-8')


def main(data: bool = False) -> None:
    '''
    Contains main analysis logic.
    Takes in boolean 'data' which represents whether the aggregate table
    values have already been calculated and saved as 'results.csv'. (avoids
    rerunning expensive calculations)

    '''
    if data is True:
        write_agg_csv()
    df = pd.read_csv('results.csv')
    # Calculate the best model for each Stock
    df['Best Model'] = df.loc[:, ['KNN', 'KNN_NO_VOLUME',
                              'FR', 'FR_NO_VOLUME']].idxmin(axis=1)
    # Print out the best Models
    print('Number of times each model was the most accurate')
    print(df['Best Model'].value_counts())
    print()
    # Calculate the worst model
    df['Worst Model'] = df.loc[:, ['KNN', 'KNN_NO_VOLUME',
                               'FR', 'FR_NO_VOLUME']].idxmax(axis=1)
    # Print out the worst models
    print('Number of times each model was the least accurate')
    print(df['Worst Model'].value_counts())
    print(df['Worst Model'])


if __name__ == '__main__':
    main()
