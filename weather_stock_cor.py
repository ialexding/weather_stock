# -*- coding: utf-8 -*-
"""
Updated on 1/14/2018
Correlation between weather and stock return

@author: Feng Ding
"""

import bs4 as bs
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from pandas_datareader._utils import RemoteDataError
import numbers
import datetime

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
    end = dt.datetime(2017, 11, 19)
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
    
            #df.rename(columns={'Adj Close':ticker + '_' + ticker_sector[ticker]}, inplace=True)
            #df.rename(columns={'Adj Close':ticker + '&' + ticker_sector[ticker]}, inplace=True)
            df.rename(columns={'Adj Close':ticker + '&' + ticker_sector[ticker].upper()}, inplace=True)
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

#-----stock close and return data frame
df_sp_close = pd.read_csv('sp500_closes.csv')
print(df_sp_close.head())
sp500_changes = pd.read_csv('sp500_changes.csv')
print(sp500_changes.head())
sp500_changes.describe()


#-----------whether data
df_weather = pd.read_csv('C:/Users/Feng/Desktop/weather_data.csv')
#-----new york data
#df_weather1 = df_weather[['DATE','REPORTTPYE','DAILYWeather','DAILYAverageDryBulbTemp','DAILYAverageRelativeHumidity','DAILYAverageStationPressure','DAILYAverageWindSpeed']].loc[df_weather['REPORTTPYE']=='SOD'].loc[df_weather['STATION_NAME']=='LA GUARDIA AIRPORT NY US']
df_weather1 = df_weather[['DATE','DAILYAverageDryBulbTemp','DAILYAverageRelativeHumidity','DAILYAverageStationPressure','DAILYAverageWindSpeed']].loc[df_weather['REPORTTPYE']=='SOD'].loc[df_weather['STATION_NAME']=='LA GUARDIA AIRPORT NY US']
df_weather1.rename(columns={'DATE':'Date'}, inplace=True)
print(df_weather1.head())
df_weather1.describe()

#-----Chicago data
df_weather2 = df_weather[['DATE','DAILYAverageDryBulbTemp','DAILYAverageRelativeHumidity','DAILYAverageStationPressure','DAILYAverageWindSpeed']].loc[df_weather['REPORTTPYE']=='SOD'].loc[df_weather['STATION_NAME']=='CHICAGO OHARE INTERNATIONAL AIRPORT IL US']
df_weather2.rename(columns={'DATE':'Date'}, inplace=True)
print(df_weather2.head())
#-----San Francisco data
df_weather3 = df_weather[['DATE','DAILYAverageDryBulbTemp','DAILYAverageRelativeHumidity','DAILYAverageStationPressure','DAILYAverageWindSpeed']].loc[df_weather['REPORTTPYE']=='SOD'].loc[df_weather['STATION_NAME']=='SAN FRANCISCO INTERNATIONAL AIRPORT CA US']
df_weather3.rename(columns={'DATE':'Date'}, inplace=True)
print(df_weather3.head())
#--------convert date to yyyy-mm-dd if needed
#for x in range(1,len(df_weather1.index)):
  #  datetime.datetime.strptime(df_weather1.iloc[x-1][0], "%m/%d/%Y").strftime("%Y-%m-%d")


#------merge stock price change  data with individual city weather data for analysis
merge_ticker_ny = pd.merge(sp500_changes, df_weather1, on='Date', how='left')
merge_ticker_ch = pd.merge(sp500_changes, df_weather2, on='Date', how='left')
merge_ticker_sf = pd.merge(sp500_changes, df_weather3, on='Date', how='left')
print(merge_ticker_sf.head())

#------------stock sector return data
df_stock_sec = pd.read_csv('sp500_changes.csv')
for x in range(1,len(df_stock_sec.columns)):
    df_stock_sec.rename(columns={df_stock_sec.columns[x] : df_stock_sec.columns[x].split("&")[1]}, inplace=True)
print(df_stock_sec.head())

#-------creat a table for sector ave return
df_sec_return = df_stock_sec.groupby(by=df_stock_sec.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
print(df_sec_return.head())

#------merge sector return with weather for analysis
merge_sector_ny = pd.merge(df_sec_return, df_weather1, on='Date', how='left')
merge_sector_ch = pd.merge(df_sec_return, df_weather2, on='Date', how='left')
merge_sector_sf = pd.merge(df_sec_return, df_weather3, on='Date', how='left')
print(merge_sector_sf.head())


#------create correlation data frame for individul stocks with three cities
stocks_ny = []
corr_tempe_ny = []
corr_humidity_ny = []
corr_preassure_ny = []
corr_wind_ny = []
for i in range(1,len(sp500_changes.columns)):
    stocks_ny.append(list(merge_ticker_ny.columns.values)[i])
    corr_tempe_ny.append(merge_ticker_ny[merge_ticker_ny.columns[i]].corr(merge_ticker_ny['DAILYAverageDryBulbTemp']))
    corr_humidity_ny.append(merge_ticker_ny[merge_ticker_ny.columns[i]].corr(merge_ticker_ny['DAILYAverageRelativeHumidity']))
    corr_preassure_ny.append(merge_ticker_ny[merge_ticker_ny.columns[i]].corr(merge_ticker_ny['DAILYAverageStationPressure']))
    corr_wind_ny.append(merge_ticker_ny[merge_ticker_ny.columns[i]].corr(merge_ticker_ny['DAILYAverageWindSpeed']))
corr_ny = pd.DataFrame(np.column_stack([stocks_ny, corr_tempe_ny, corr_humidity_ny,corr_preassure_ny,corr_wind_ny ]), 
                               columns=['stocks', 'corr_tempe_ny', 'corr_humidity_ny','corr_preassure_ny','corr_wind_ny'])

print(corr_ny.head())

stocks_ch = []
corr_tempe_ch = []
corr_humidity_ch = []
corr_preassure_ch = []
corr_wind_ch = []
for i in range(1,len(sp500_changes.columns)):
    stocks_ch.append(list(merge_ticker_ch.columns.values)[i])
    corr_tempe_ch.append(merge_ticker_ch[merge_ticker_ch.columns[i]].corr(merge_ticker_ch['DAILYAverageDryBulbTemp']))
    corr_humidity_ch.append(merge_ticker_ch[merge_ticker_ch.columns[i]].corr(merge_ticker_ch['DAILYAverageRelativeHumidity']))
    corr_preassure_ch.append(merge_ticker_ch[merge_ticker_ch.columns[i]].corr(merge_ticker_ch['DAILYAverageStationPressure']))
    corr_wind_ch.append(merge_ticker_ch[merge_ticker_ch.columns[i]].corr(merge_ticker_ch['DAILYAverageWindSpeed']))
corr_ch = pd.DataFrame(np.column_stack([stocks_ch, corr_tempe_ch, corr_humidity_ch,corr_preassure_ch,corr_wind_ch ]), 
                               columns=['stocks', 'corr_tempe_ch', 'corr_humidity_ch','corr_preassure_ch','corr_wind_ch'])
print(corr_ch.head())

stocks_sf = []
corr_tempe_sf = []
corr_humidity_sf = []
corr_preassure_sf = []
corr_wind_sf = []
for i in range(1,len(sp500_changes.columns)):
    stocks_sf.append(list(merge_ticker_sf.columns.values)[i])
    corr_tempe_sf.append(merge_ticker_sf[merge_ticker_sf.columns[i]].corr(merge_ticker_sf['DAILYAverageDryBulbTemp']))
    corr_humidity_sf.append(merge_ticker_sf[merge_ticker_sf.columns[i]].corr(merge_ticker_sf['DAILYAverageRelativeHumidity']))
    corr_preassure_sf.append(merge_ticker_sf[merge_ticker_sf.columns[i]].corr(merge_ticker_sf['DAILYAverageStationPressure']))
    corr_wind_sf.append(merge_ticker_sf[merge_ticker_sf.columns[i]].corr(merge_ticker_sf['DAILYAverageWindSpeed']))
corr_sf = pd.DataFrame(np.column_stack([stocks_sf, corr_tempe_sf, corr_humidity_sf,corr_preassure_sf,corr_wind_sf ]), 
                               columns=['stocks', 'corr_tempe_sf', 'corr_humidity_sf','corr_preassure_sf','corr_wind_sf'])
print(corr_sf.head())

#------------creat sector correlation data frame
stocks_ny_sec = []
corr_tempe_ny_sec = []
corr_humidity_ny_sec = []
corr_preassure_ny_sec = []
corr_wind_ny_sec = []
for i in range(1,len(df_sec_return.columns)):
    stocks_ny_sec.append(list(merge_sector_ny.columns.values)[i])
    corr_tempe_ny_sec.append(merge_sector_ny[merge_sector_ny.columns[i]].corr(merge_sector_ny['DAILYAverageDryBulbTemp']))
    corr_humidity_ny_sec.append(merge_sector_ny[merge_sector_ny.columns[i]].corr(merge_sector_ny['DAILYAverageRelativeHumidity']))
    corr_preassure_ny_sec.append(merge_sector_ny[merge_sector_ny.columns[i]].corr(merge_sector_ny['DAILYAverageStationPressure']))
    corr_wind_ny_sec.append(merge_sector_ny[merge_sector_ny.columns[i]].corr(merge_sector_ny['DAILYAverageWindSpeed']))
corr_ny_sec= pd.DataFrame(np.column_stack([stocks_ny_sec, corr_tempe_ny_sec, corr_humidity_ny_sec,corr_preassure_ny_sec,corr_wind_ny_sec ]), 
                               columns=['sector', 'corr_tempe_ny_sec', 'corr_humidity_ny_sec','corr_preassure_ny_sec','corr_wind_ny_sec'])

print(corr_ny_sec.head())

stocks_ch_sec = []
corr_tempe_ch_sec = []
corr_humidity_ch_sec = []
corr_preassure_ch_sec = []
corr_wind_ch_sec = []
for i in range(1,len(df_sec_return.columns)):
    stocks_ch_sec.append(list(merge_sector_ch.columns.values)[i])
    corr_tempe_ch_sec.append(merge_sector_ch[merge_sector_ch.columns[i]].corr(merge_sector_ch['DAILYAverageDryBulbTemp']))
    corr_humidity_ch_sec.append(merge_sector_ch[merge_sector_ch.columns[i]].corr(merge_sector_ch['DAILYAverageRelativeHumidity']))
    corr_preassure_ch_sec.append(merge_sector_ch[merge_sector_ch.columns[i]].corr(merge_sector_ch['DAILYAverageStationPressure']))
    corr_wind_ch_sec.append(merge_sector_ch[merge_sector_ch.columns[i]].corr(merge_sector_ch['DAILYAverageWindSpeed']))
corr_ch_sec= pd.DataFrame(np.column_stack([stocks_ch_sec, corr_tempe_ch_sec, corr_humidity_ch_sec,corr_preassure_ch_sec,corr_wind_ch_sec ]), 
                               columns=['stocks', 'corr_tempe_ch_sec', 'corr_humidity_ch_sec','corr_preassure_ch_sec','corr_wind_ch_sec'])
print(corr_ch_sec.head())

stocks_sf_sec = []
corr_tempe_sf_sec = []
corr_humidity_sf_sec = []
corr_preassure_sf_sec = []
corr_wind_sf_sec = []
for i in range(1,len(df_sec_return.columns)):
    stocks_sf_sec.append(list(merge_sector_sf.columns.values)[i])
    corr_tempe_sf_sec.append(merge_sector_sf[merge_sector_sf.columns[i]].corr(merge_sector_sf['DAILYAverageDryBulbTemp']))
    corr_humidity_sf_sec.append(merge_sector_sf[merge_sector_sf.columns[i]].corr(merge_sector_sf['DAILYAverageRelativeHumidity']))
    corr_preassure_sf_sec.append(merge_sector_sf[merge_sector_sf.columns[i]].corr(merge_sector_sf['DAILYAverageStationPressure']))
    corr_wind_sf_sec.append(merge_sector_sf[merge_sector_sf.columns[i]].corr(merge_sector_sf['DAILYAverageWindSpeed']))
corr_sf_sec= pd.DataFrame(np.column_stack([stocks_sf_sec, corr_tempe_sf_sec, corr_humidity_sf_sec,corr_preassure_sf_sec,corr_wind_sf_sec ]), 
                               columns=['stocks', 'corr_tempe_sf_sec', 'corr_humidity_sf_sec','corr_preassure_sf_sec','corr_wind_sf_sec'])



#--------------------------chart:individual stocks
plt.plot(corr_ny.loc[:,"corr_tempe_ny"].astype(float).round(1),corr_ny.loc[:,"corr_humidity_ny"].astype(float).round(1), 'b-')
plt.plot(corr_ny.loc[:,"corr_preassure_ny"].astype(float).round(1),corr_ny.loc[:,"corr_wind_ny"].astype(float).round(1),'g-')

plt.plot(corr_ch.loc[:,"corr_tempe_ch"].astype(float).round(1),corr_ch.loc[:,"corr_humidity_ch"].astype(float).round(1), 'b-')
plt.plot(corr_ch.loc[:,"corr_preassure_ch"].astype(float).round(1),corr_ch.loc[:,"corr_wind_ch"].astype(float).round(1),'g-')

plt.plot(corr_sf.loc[:,"corr_tempe_sf"].astype(float).round(1),corr_sf.loc[:,"corr_humidity_sf"].astype(float).round(1), 'b-')
plt.plot(corr_sf.loc[:,"corr_preassure_sf"].astype(float).round(1),corr_sf.loc[:,"corr_wind_sf"].astype(float).round(1),'g-')

plt.plot(corr_ny_sec.loc[:,"corr_tempe_ny_sec"].astype(float).round(1),corr_ny_sec.loc[:,"corr_humidity_ny_sec"].astype(float).round(1), 'b-')
plt.plot(corr_ny_sec.loc[:,"corr_preassure_ny_sec"].astype(float).round(1),corr_ny_sec.loc[:,"corr_wind_ny_sec"].astype(float).round(1),'g-')


plt.plot(corr_ch_sec.loc[:,"corr_tempe_ch_sec"].astype(float).round(1),corr_ch_sec.loc[:,"corr_humidity_ch_sec"].astype(float).round(1), 'b-')
plt.plot(corr_ch_sec.loc[:,"corr_preassure_ch_sec"].astype(float).round(1),corr_ch_sec.loc[:,"corr_wind_ch_sec"].astype(float).round(1),'g-')

plt.plot(corr_sf_sec.loc[:,"corr_tempe_sf_sec"].astype(float).round(1),corr_sf_sec.loc[:,"corr_humidity_sf_sec"].astype(float).round(1), 'b-')
plt.plot(corr_sf_sec.loc[:,"corr_preassure_sf_sec"].astype(float).round(1),corr_sf_sec.loc[:,"corr_wind_sf_sec"].astype(float).round(1),'g-')

plt.show()

"""
next steps:
    1-build functions where applicable
    2-more regressional analysis

"""


#-------------------------------testing 
----------chart test
x = [corr_sf_sec[2:]]
corr_sf_sec.astype(float).round(1)
print(corr_sf_sec.loc[:,"corr_tempe_sf_sec"].astype(float).round(2))
corr_sf_sec.loc[:,"corr_tempe_sf_sec"].round()
print(merge_ticker_sf.head())
print(merge_sector_sf.head())



import matplotlib.pyplot as plt
plt.scatter(corr_sf_sec['corr_tempe_sf_sec'], corr_sf_sec['corr_humidity_sf_sec'])
plt.show()
plt.hist( corr_sf_sec['corr_humidity_sf_sec'],10,normed=1,alpha=0.75)

plt.scatter(corr_sf_sec)
plt.show()
plt.show()

----------------------regression analysis-------------




-----------corr test
# Import `PCA` from `sklearn.decomposition`
from sklearn.decomposition import PCA
# Build the model
pca = PCA(n_components=2)

# Reduce the data, output is ndarray
reduced_data = pca.fit_transform(merge_ticker_sf.columns[1:])

# Inspect the shape of `reduced_data`
reduced_data.______

# print out the reduced data
print(_______________)



