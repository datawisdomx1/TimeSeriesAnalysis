#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:01:41 2022

@author: nitinsinghal
"""

# Time Series analysis for order data - ARIMA acf pacf adfuller decompose 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

# Load the data
all_data = pd.read_csv('./Data.csv')

# Perform EDA - see the data types, content, statistical properties
print(all_data.describe())
print(all_data.info())
print(all_data.head(5))
print(all_data.dtypes)

# Perform data wrangling - remove duplicate values and set null values to 0
all_data.drop_duplicates(inplace=True)

# Only used fillna=0. Dropna not used as other columns/rows have useful data
all_data.fillna(0, inplace=True)

# Convert dates to year-week no
all_data['orderdate'] = pd.to_datetime(all_data['PO_ORD'])

Ord_data = all_data[['orderdata']] 
Rcv_data = all_data[['orderdata']] 


data_Wkly = data.resample('W').sum()

# Moving Average rolling - weekly - 4 weeks
data_Wkly.head()
print(data_Wkly.dtypes)
rolling_stddev =  data_Wkly.rolling(4).std()
rolling_mean = data_Wkly.rolling(4).mean()
fig, ax = plt.subplots()
ax.plot(rolling_stddev, color='blue', label='12Week MA StdDev')
ax.plot(rolling_mean, color='red', label='12Week MA Mean')
ax.plot(data_Wkly, color='green', label='Order Qty')
plt.title('TSA')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.ylim(0,600000)
plt.legend()
plt.tight_layout()
plt.show()

# Dickey-Fuller test for stationarity (No trend, seasonality, cyclicality, irregularity)
# Null hypothesis: Series is non-stationary. If p-value > 0.05, cannot reject Ho,
# Else, accept Ha: Series is stationary
data_Wkly_series = data_Wkly.iloc[:, 0].values
dickyfulltest = adfuller(data_Wkly_series, autolag='AIC')
print('Dickey-Fuller test for stationarity Results:')
print('test statistic: %.4f, pvalue: %.4f'%(dickyfulltest[0],dickyfulltest[1]))

# Remove trend and seasonality
# Differencing - difference between time pe - Weekly datariod t and t-1. Only removes trend.
dataWkly_diff = data_Wkly - data_Wkly.shift(-1)
dataWkly_diff_stddev =  data_Wkly.rolling(4).std()
dataWkly_diff_mean = data_Wkly.rolling(4).mean()
fig, ax = plt.subplots()
ax.plot(dataWkly_diff, color='blue', label='WklyDiff')
ax.plot(dataWkly_diff_mean, color='green', label='WklyDiff4WMA')
ax.plot(dataWkly_diff_stddev, color='red', label='WklyDiff4WStdDev')
plt.title('TSA t,t-1')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.ylim(-300000,500000)
plt.legend()
plt.tight_layout()
plt.show()

# Decomposition - seasonal_decompose. Removes trend and seasonality
decompose = seasonal_decompose(data_Wkly)
trend = decompose.trend
seasonality = decompose.seasonal
residual = decompose.resid

plt.subplot(411)
plt.plot(data_Wkly, label='Weekly data')
plt.legend(loc='right')
plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='right')
plt.subplot(413)
plt.plot(seasonality, label='seasonality')
plt.legend(loc='right')
plt.subplot(414)
plt.plot(residual, label='residual')
plt.legend(loc='right')
plt.title('TSA Decomposition')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.tight_layout()
plt.show()

# decompose the residual error as seasonality/trend have been removed
residual.dropna(inplace=True)
residualma = residual.rolling(4).mean()
residualstd = residual.rolling(4).std()
plt.figure(figsize=(10,5))
plt.plot(residual, color='blue', label='Residual')
plt.plot(residualma, color='green', label='ResidualMA')
plt.plot(residualstd, color='red', label='ResidualStdDev')
plt.title('TSA Residual')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.legend()
plt.tight_layout()
plt.show()

# check for stationarity in residual error
residualseries = residual
residualdickyfulltest = adfuller(residualseries, autolag='AIC')
print('test statistic: ', residualdickyfulltest[0], 'pvalue: ', residualdickyfulltest[1])

# Forecasting time series
# ARIMA. p- , d- , q-
#ACF and PACF plots:
dataWkly_diff.dropna(inplace=True)
lag_acf = acf(data_Wkly, nlags=4)
lag_pacf = pacf(data_Wkly, nlags=4, method='ols')
#Plot ACF -q : 
plt.figure(figsize=(10,5))
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_Wkly)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_Wkly)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF -p :
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_Wkly)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_Wkly)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#The p,d,q values can be specified using the order argument of ARIMA 
# which take a tuple (p,d,q)

# AR model
# AR - autoregression (AR) method models the next step in the sequence as a linear function of 
# the observations at prior time steps. Suitable for univariate time series without trend and 
# seasonal components.

# MA model
# MA - moving average (MA) method models the next step in the sequence as a linear function of 
# the residual errors from a mean process at prior time steps. The notation for the model involves 
# specifying the order of the model q as a parameter to the MA function, e.g. MA(q)

# ARIMA - The Autoregressive Integrated Moving Average (ARIMA) method models the next step in
# the sequence as a linear function of the differenced observations and residual errors at prior 
# time steps. It combines both Autoregression (AR) and Moving Average (MA) models as well as a 
# differencing pre-processing step of the sequence to make the sequence stationary, called 
# integration (I). The notation for the model involves specifying the order for the AR(p), I(d), 
# and MA(q) models as parameters to an ARIMA function, e.g. ARIMA(p, d, q)

# ARIMA model
model = ARIMA(data_Wkly, order=(2, 1, 2))  
result_ARIMA = model.fit(disp=-1)  
plt.figure(figsize=(10,5))
plt.plot(data_Wkly)
plt.plot(result_ARIMA.fittedvalues, color='red')

result_ARIMA.summary()

ARIMA_predict = result_ARIMA.predict()

# Predict for next 2 years, using the 2 years provided. 2*52+2*52 = 208
result_ARIMA.plot_predict(start=1,end=208) # start, end

### Using Logarithms ####
# Remove trend by taking logarithm of data for making time series stationary
data_Wklylog = np.log(data_Wkly)
data_Wklylogma = data_Wklylog.rolling(4).mean()
fig, ax = plt.subplots()
ax.plot(data_Wklylog, color='blue', label='WklyLog')
ax.plot(data_Wklylogma, color='green', label='WklyLog 12WeekMA')
plt.title('TSALog Order Qty')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.ylim(0,15)
plt.legend()
plt.tight_layout()
plt.show()

# Remove trend and seasonality - Log weekly data
# Differencing - difference between time period t and t-1. Only removes trend.
datalogdiff = data_Wklylog - data_Wklylog.shift()
datalogdiffma = datalogdiff.rolling(4).mean()
datalogdiffstd = datalogdiff.rolling(4).std()
fig, ax = plt.subplots()
ax.plot(datalogdiff, color='blue', label='WklyLogDiff')
ax.plot(datalogdiffma, color='green', label='WklyLogDiff12WMA')
ax.plot(datalogdiffstd, color='red', label='WklyLogDiff12WStdDev')
plt.title('WklyLog Difference t,t-1')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.ylim(-10,5)
plt.legend()
plt.tight_layout()
plt.show()

# Decomposition - seasonal_decompose. Removes trend and seasonality
decompose = seasonal_decompose(data_Wklylog)
trend = decompose.trend
seasonal = decompose.seasonal
residual = decompose.resid

plt.subplot(411)
plt.plot(data_Wklylog, label='WklyLog data')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='residual')
plt.legend(loc='best')
plt.tight_layout()

# decompose the residual error as seasonality/trend have been removed
residual.dropna(inplace=True)
residualma = residual.rolling(12).mean()
residualstd = residual.rolling(12).std()
plt.figure(figsize=(10,5))
plt.plot(residual, color='blue', label='Residual')
plt.plot(residualma, color='red', label='ResidualMA')
plt.plot(residualstd, color='green', label='ResidualStdDev')
plt.title(' WklyLog Residual')
plt.ylabel('WklyLog Total Order Qty')
plt.xlabel('Year-Week')
plt.legend()
plt.tight_layout()
plt.show()

# check for stationarity in residual error
residualseries = residual
residualdickyfulltest = adfuller(residualseries, autolag='AIC')
print('test statistic: ', residualdickyfulltest[0], 'pvalue: ', residualdickyfulltest[1])

# Forecasting time series
# ARIMA. p- , d- , q-
#ACF and PACF plots:
data_Wklylog.dropna(inplace=True)
lag_acf = acf(data_Wklylog, nlags=12)
lag_pacf = pacf(data_Wklylog, nlags=12, method='ols')
#Plot ACF -q : 
plt.figure(figsize=(10,5))
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_Wklylog)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_Wklylog)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF -p :
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_Wklylog)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_Wklylog)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#The p,d,q values can be specified using the order argument of ARIMA 
# which take a tuple (p,d,q)

# ARIMA model - weekly log data
model_log = ARIMA(data_Wklylog, order=(1, 0, 1))  
results_ARIMA_log = model_log.fit(disp=-1)  
plt.figure(figsize=(10,5))
plt.plot(data_Wklylog)
plt.plot(results_ARIMA_log.fittedvalues, color='red')

results_ARIMA_log.summary()

ARIMA_predict = results_ARIMA_log.predict()

# Predict for next 2 years, using the 2 years provided. 2*52+2*52 = 208
results_ARIMA_log.plot_predict(start=1,end=208) # start, end

### Using Exponential Moving Average - EMA ####
# EMA with smoothing factor. Filters noise, gives more weight to recent data using decay factor
ema01 =  data_Wkly.ewm(alpha=0.1).mean()
ema03 = data_Wkly.ewm(alpha=0.3).mean()
fig, ax = plt.subplots()
ax.plot(ema01, color='blue', label='ExpMA alpha=0.1')
ax.plot(ema03, color='red', label='ExpMA alpha=0.3')
ax.plot(data_Wkly, color='green', label='Order Qty')
plt.title('Weekly Exponential Order Qty')
plt.ylabel('Weekly Total Order Qty')
plt.xlabel('Year-Week')
plt.ylim(0,600000)
plt.legend()
plt.tight_layout()
plt.show()










