#!/usr/bin/env python
# coding: utf-8

# # Stock Prediction
# #### This project introduces common techniques to manipulate time series and make predictions.
# ### Goal: To predict the closing price in the next five trading days starting February 1st, 2019.
# 
# 
# The data is a sample from the historical end of day (EOD) stock prices for all US public companies excluding over the counter (OTC) stocks. Open, high, low, close, volume, as well as prices adjusted for splits and dividends, are included. Includes delisted securities over the 5 year time period. History is daily back to January 1st, 2013.
# https://about.intrinio.com/bulk-financial-data-downloads

# ### Cleaning data
# 1. Import libraries for our analysis. 
# 2. Define the mean average percentage error (MAPE), this will be our error metric.

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook

from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


#import our dataset
DATAPATH = 'data/stock_prices_sample.csv'


# In[73]:


data = pd.read_csv(DATAPATH, index_col=['DATE'], parse_dates=['DATE'])
data.head(10)


# ###### Here we have few entries that are different stock than the New Germany Fund (GF). Also, we have an entry of intraday information, but we only want end of day (EOD) information.

# In[74]:


data.shape


# In[75]:


data.dtypes


# In[76]:


#Let’s get rid of the unwanted entries:
data = data[data.TICKER != 'GEF']
data = data[data.TYPE != 'Intraday']


# In[77]:


data.head()


# In[78]:


#remove unwanted columns, focus on the stock’s closing price only:
drop_cols = ['SPLIT_RATIO', 'EX_DIVIDEND', 'ADJ_FACTOR', 'ADJ_VOLUME', 'ADJ_CLOSE', 'ADJ_LOW', 'ADJ_HIGH', 'ADJ_OPEN', 'VOLUME', 'FREQUENCY', 'TYPE', 'FIGI', 'SECURITY_ID']
data.drop(drop_cols, axis=1, inplace=True)


# In[79]:


data.head()


#  ### Exploratory data analysis (EDA)

# In[80]:


#Let’s see what the closing price looks like:
plt.figure(figsize=(17, 8))
plt.plot(data.CLOSE)
plt.title('Closing price of Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()


# Clearly, we see that this is not a stationary process, and it is hard to tell if there is some kind of seasonality.

# ### Moving average
# We use the moving average model to smooth our time series. For that, the helper function will run the moving average model on a specified time window and it will plot the result smoothed curve:

# In[81]:


def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
    #Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
            
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)


# #### Smooth by the previous 5 days (by the work week)
# 

# In[82]:


plot_moving_average(data.CLOSE, 5)


# ###### Here we can hardly see a trend, because it is too close to actual curve. Let’s see the result of smoothing by the previous month, and previous quarter.

# #### Smooth by the previous month (30 days)

# In[83]:



plot_moving_average(data.CLOSE, 30)


# #### Smooth by previous quarter (90 days)

# In[84]:


plot_moving_average(data.CLOSE, 90, plot_intervals=True)


# ##### Trends are easier to spot now. The 30-day and 90-day trend show a downward curve at the end. This might mean that the stock is likely to go down in the following days.

# ### Exponential smoothing
# to see if it can pick up a better trend

# In[85]:


def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


# In[86]:


def plot_exponential_smoothing(series, alphas):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True);


# In[87]:


plot_exponential_smoothing(data.CLOSE, [0.05, 0.3])

#we use 0.05 and 0.3 as values for the smoothing factor. 
#Feel free to try other values and see what the result is.


# As you can see, an alpha value of 0.05 smoothed the curve while picking up most of the upward and downward trends.

# ### Double exponential smoothing

# In[88]:


def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


# In[89]:


def plot_double_exponential_smoothing(series, alphas, betas):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series.values, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing")
    plt.grid(True)


# In[90]:


plot_double_exponential_smoothing(data.CLOSE, alphas=[0.9, 0.02], betas=[0.9, 0.02])


# You may experiment with different alpha and beta combinations to get better looking curves.

# ### Stationarity
# We must turn our series into a stationary process in order to model it. Stationarity refers to the stability of the mean. There is no trend and there is stability of the correlation factor, in other words, structure of the correlation remains constant over time, while in a non stationarity series the trend and correlation dont remain constant. We apply the Dickey-Fuller test to see if it is a stationary process:

# In[91]:


def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
tsplot(data.CLOSE, lags=30)


# By using the Dickey-Fuller test, we see the time series is unsurprisingly non-stationary. Also, looking at the autocorrelation plot, we see that it is very high, and it seems that there is no clear seasonality.
# 
# Therefore, to get rid of the high autocorrelation and to make the process stationary, we take the first difference. We simply take the time series itself with a lag of one day:

# In[92]:


data_diff = data.CLOSE - data.CLOSE.shift(1)

tsplot(data_diff[1:], lags=30)


# ### SARIMA
# Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component.
# It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality. more here: https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
# 
# SARIMA, we will define a few parameters and a range of values for other parameters to generate a list of all possible combinations of p, q, d, P, Q, D, s:

# In[93]:


#Set initial values and some bounds
ps = range(0, 5)
d = 1
qs = range(0, 5)
Ps = range(0, 5)
D = 1
Qs = range(0, 5)
s = 5

#Create a list with all possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# We will try each combination (from all 625) and train SARIMA with each so to find the best performing model. (it will take some time)

# In[94]:


def optimize_SARIMA(parameters_list, d, D, s):
    
    """    
    Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    
    
    results = []
    best_aic = float('inf')
    
    for param in tqdm_notebook(parameters_list):
        try: model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(param[0], d, param[1]),
                                               seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        
        
    #Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    
    
    #Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table

result_table = optimize_SARIMA(parameters_list, d, D, s)


# In[98]:


#Set parameters that give the lowest AIC (Akaike Information Criteria)
#AIC deals with both the risk of overfitting and the risk of underfitting.

p, q, P, Q = result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)

print(best_model.summary())


# ##### Now, we can predict the closing price of the next five trading days and evaluate the MAPE of the model:

# In[99]:


def plot_SARIMA(series, model, n_steps):
    """
        Plot model vs predicted values
        
        series - dataset with time series
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    """
    
    data = series.copy().rename(columns = {'CLOSE': 'actual'})
    data['arima_model'] = model.fittedvalues
    #Make a shift on s+d steps, because these values were unobserved by the model due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    #Forecast on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    #Calculate error
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    
    plt.figure(figsize=(17, 8))
    plt.title('Mean Absolute Percentage Error: {0:.2f}%'.format(error))
    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1],alpha=0.5, color='lightgrey')
    plt.plot(data, label='actual')
    plt.legend()
    plt.grid(True);

    
# Now, we can predict the closing price of the next five trading days and evaluate the MAPE of the model:
# plot_SARIMA(data, best_model, 5)
print(best_model.predict(start=data.CLOSE.shape[0], end=data.CLOSE.shape[0] + 5))
print(mean_absolute_percentage_error(data.CLOSE[s+d:], best_model.fittedvalues[s+d:]))


# In this case, we have a MAPE of 78.5%, which is very good!
# 
# #### Now, to compare our prediction with actual data, I took financial data from https://ca.finance.yahoo.com/quote/GF/history?p=GF and created a dataframe:

# In[100]:


comparison = pd.DataFrame({'actual': [13.42, 13.49, 13.56, 13.61, 13.38, 13.21],
                          'predicted': [18.96, 18.97, 18.96, 18.92, 18.94, 18.92]}, 
                          index = pd.date_range(start='2019-02-01', periods=6,))


# In[101]:


comparison.head()


# In[102]:


plt.figure(figsize=(17, 8))
plt.plot(comparison.actual)
plt.plot(comparison.predicted)
plt.title('Predicted closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.legend(loc='best')
plt.grid(False)
plt.show()


# It seems like we are a bit off in our predictions. In fact, we didn't missed an opportunity to make money, since our predictions result in a net gain, whereas the actual closing prices show a net loss. Time to invest, people!
# 

# Credit to  Peters Morgan, Edureca Co & Marco Peixeiro for teaching me common techniques to manipulate time series and make predictions.
