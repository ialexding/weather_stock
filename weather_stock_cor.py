# -*- coding: utf-8 -*-
"""
Updated on 12/2/2017
Correlation between weather and stock return

@author: Feng Ding
"""

import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from pandas_datareader._utils import RemoteDataError

style.use('ggplot')

#complie S&P 500 stock tickers from wikipedia
def sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        #ticker = row.findAll('td')[0].text
        ticker = str(row.findAll('td')[0].string.strip())
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers
#sp500_tickers()

#combine ticker sector list
def sp500_ticker_sector():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    sectors = []
    ticker_sector = dict()
    
    for row in table.findAll('tr')[1:]:
        #ticker = row.findAll('td')[0].text
        ticker = str(row.findAll('td')[0].string.strip())
        sector = str(row.findAll('td')[3].string.strip()).lower().replace(' ', '_')
        #if ticker not in ticker_sector:
        #    ticker_sector[ticker] = list()
        #ticker_sector[ticker].append(sector)
        tickers.append(ticker)
        sectors.append(sector)
    ticker_sector = dict(zip(tickers,sectors))
            
    with open("sp500ticker_sector.pickle","wb") as f:
        pickle.dump(ticker_sector,f)
        
    return ticker_sector
sp500_ticker_sector()

#pull individual stock data from yahoo and save by tickers for compiling
def yahoo_data(reload_sp500=False):
    
    start = dt.datetime(2016, 11, 1)
    end = dt.datetime(2016, 12, 31)
    errors = 0
    count = 0
    error_tickers = []
    
    if reload_sp500:
        tickers = sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
        
    for ticker in tickers:
    # due to yahoo connection issues, code need multiple try to pull data
    # and in case connection breaks, it's good save our progress
        try:   
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                #df = web.DataReader(ticker, "yahoo", start, end)
                df = web.DataReader(ticker.strip('\n'), "yahoo", start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
               
            else:
                print('Already saved {}'.format(ticker))
                count = count + 1
        except RemoteDataError:
            print("No connection for ticker '%s'" % ticker)
            if ticker not in error_tickers:
                error_tickers.append(ticker)
            errors = len(error_tickers)
            continue
    print('tickers count: ',count)
    #three tickers: BHF, BF.B, BRK.B causing errors
    while errors > 3:    
        for ticker1 in error_tickers:
            try:
                if not os.path.exists('stock_dfs/{}.csv'.format(ticker1)):
                    #df = web.DataReader(ticker, "yahoo", start, end)
                    df = web.DataReader(ticker1.strip('\n'), "yahoo", start, end)
                    df.to_csv('stock_dfs/{}.csv'.format(ticker1))
                    error_tickers.remove(ticker1)
                    count = count + 1
                    
                else:
                    print('Already saved {}'.format(ticker1))
                    
            except RemoteDataError:
                print("No connection for ticker '%s'" % ticker1)
                if ticker1 not in error_tickers:
                    error_tickers.append(ticker1)
                print('connection error count: ', len(error_tickers), ' - ', ticker1)
                continue
        errors = len(error_tickers)
    print ('number of tickers successfully connected: ', count)
    print('tickers with connection issues: ', error_tickers)             

yahoo_data()

#compile individual stock data into one data frame
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
        
    with open("sp500ticker_sector.pickle","rb") as f:
        ticker_sector = pickle.load(f)

    main_df = pd.DataFrame()
    noData_tickers = []
    
    for count,ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
    
            df.rename(columns={'Adj Close':ticker + '_' + ticker_sector[ticker]}, inplace=True)
            df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
    
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        except Exception:
                if ticker not in noData_tickers:
                    noData_tickers.append(ticker)
                continue
        if count % 10 == 0:
            print('ticker compiling count: ', count)
    #calculate stock price change
    change_df = main_df.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.

    print('total number of tickers: ', len(tickers))
    print('no data tickers: ', noData_tickers)
    print('top rows of stock close price data:')
    print(main_df.head())
    print('top rows of stock price change data:')
    print(change_df.head())
    
    main_df.to_csv('sp500_closes.csv')
    change_df.to_csv('sp500_changes.csv')
    #main_df.to_csv("/home/feng/change_df.csv")
compile_data()
"""
next steps:
    1-figure out how to pull sector change 
    2-join weather data

"""