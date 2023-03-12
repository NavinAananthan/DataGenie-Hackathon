import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,kpss
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def check_null(data):
    null=data.isnull().sum()
    for index,value in null.items():
        if value!=0:
            new_data=data.fillna(method='bfill')
        else:
            new_data=data

    return new_data
        
    
    
def findTrend(df):

    autocorr = acf(df['Value'], nlags=len(df)-1)
    lag = np.argmax(np.abs(autocorr[1:]) >= 1.96) + 1
    if lag > 0:
        bool1=True
        #print("Trend detected with a period of", lag, "time steps")
    else:
        bool1=False
        #print("No trend detected")

    decomposition = seasonal_decompose(df['Value'], model='additive', period=12)
    trend = decomposition.trend
    if trend is not None:
        bool2=True
        # print("Trend detected")
    else:
        bool2=False
        # print("No trend detected")
    
    return bool1 and bool2

def adfullerTest(value):
    result_adf = adfuller(value)
    p_value_adf = result_adf[1]
    return p_value_adf

def kpssTest(value):
    result_kpss = kpss(value)
    p_value_kpss = result_kpss[1]
    return p_value_kpss

def isSeasonal(df):
    # perform ADF test
    adf_val=adfullerTest(df['Value'])

    # perform KPSS test
    kpss_val=kpssTest(df['Value'])

    # check if p-values are below significance level of 0.05
    if adf_val < 0.05 and kpss_val > 0.05:
        seasonal=True
        # print('The time series data is seasonal.')
    else:
        seasonal=False
        # print('The time series data is not seasonal.')

    return seasonal


def isStationary(df):

    adf_val=adfullerTest(df['Value'])

    # perform KPSS test
    kpss_val=kpssTest(df['Value'])

    # check if p-values are below significance level of 0.05
    if adf_val < 0.05 and kpss_val < 0.05:
        stationary=True
        # print('The time series data is seasonal.')
    else:
        stationary=False
        # print('The time series data is not seasonal.')

    return stationary

def findACF(df):
    # Compute the autocorrelation coefficients for different lag values
    autocorr = []
    for lag in range(1, len(df)):
        corr = df['Value'].autocorr(lag=lag)
        autocorr.append(corr)

    # Print the autocorrelation coefficients for each lag
    for lag, corr in enumerate(autocorr):
        print(f"Lag {lag+1}: {corr}")
        
        

def check_stationarity(data):

    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[1] > 0.05:
        return 0
    else:
        return 1



def transform(df):
    ts_log = np.log(df)
    ts_diff = ts_log.diff()
    ts_diff.dropna(inplace=True)
    return ts_log, ts_diff


def retransform(ts_log,ts_diff):
    inv_diff=ts_log.shift(1)+ts_diff
    org=np.exp(inv_diff)
    return org


    
def check_seasonality(data):

    # Perform the ADF test to check for stationarity
    adf_result = adfuller(data['point_value'])
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    print('Critical Values:', adf_result[4])

    # Perform the KPSS test to check for trend stationarity
    kpss_result = kpss(data['point_value'])
    print('KPSS Statistic:', kpss_result[0])
    print('p-value:', kpss_result[1])
    print('Critical Values:', kpss_result[3])

    # Compare the results of the ADF and KPSS tests
    if adf_result[1] < 0.05 and kpss_result[1] < 0.05:
        if adf_result[0] < adf_result[4]['5%']:
            return 0
        else:
            return 0
    elif adf_result[1] < 0.05 and kpss_result[1] >= 0.05:
        if adf_result[0] < adf_result[4]['5%']:
            return 0
        else:
            return 0
    elif adf_result[1] >= 0.05 and kpss_result[1] < 0.05:
        if kpss_result[0] < kpss_result[3]['5%']:
            return 0
        else:
            return 1
    else:
        return 0



def SARIMA(df):
    
    ts_diff=df['point_value']
    if(check_stationarity(df)==0):
        ts_log, ts_diff=transform(df['point_value'])

    acf_vals, confint = sm.tsa.acf(data['point_value'], nlags=50, alpha=0.05)
    pacf_vals = sm.tsa.pacf(data['point_value'], nlags=50)

    # find the lags where the ACF or PACF values are above the significance threshold
    acf_lags = [(i+1) for i in range(len(acf_vals)) if abs(acf_vals[i]) >= confint[i, 1]]
    pacf_lags = [(i+1) for i in range(len(pacf_vals)) if abs(pacf_vals[i]) >= confint[i, 1]]

    print(acf_lags)
    print(pacf_lags)

    d = 0
    df_diff = data.diff().dropna()
    while True:
        result = adfuller(df_diff['point_value'].diff().dropna())
        pvalue = result[1]
        if pvalue < 0.05:
            break
        d += 1
        df_diff = df_diff.diff().dropna()

    train=ts_diff.iloc[:80,:]
    test=ts_diff.iloc[80:,:]

    model = sm.tsa.statespace.SARIMAX(train, order=(pacf_lags, d, acf_lags), seasonal_order=(pacf_lags, d, acf_lags,12))
    result=model.fit()
    predict=result.forecast(steps=len(test))
    predictions_retransformed = np.exp(predict + train.iloc[-1]) # last value of training set
    predictions_retransformed_diff = predictions_retransformed.diff()
    predictions_retransformed_diff[0] += df['point_value'].iloc[79]
    for i in range(1, len(predictions_retransformed_diff)):
        predictions_retransformed_diff[i] += predictions_retransformed_diff[i-1]

    test['predicted_values'] = predictions_retransformed_diff

    plt.plot(data['original_column'], label='Original')
    plt.plot(test['predicted_values'], label='Predicted')
    plt.legend()
    plt.show()
    

       

        

data=pd.read_csv("sample_9.csv")
data=data.drop(data.columns[0],axis=1)

data['point_timestamp']=pd.to_datetime(data['point_timestamp'],infer_datetime_format=True)
data=data.set_index(['point_timestamp'])

data=check_null(data)
#print(data)

seasonal=check_seasonality(data)

if(seasonal==1):
    SARIMA(data)

#plt.plot(data)
#plt.show()
