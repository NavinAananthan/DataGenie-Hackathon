import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import os


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


def missing_value(df):
    df_ffill = df.ffill()

    return df_ffill


# Load time series data into a Pandas dataframe
# df = pd.read_csv('sample_9.csv')
# df=pd.read_csv('sample_9.csv')
# df=df.drop(df.columns[0],axis=1)
# df['point_timestamp']=pd.to_datetime(df['point_timestamp'],infer_datetime_format=True)

# df.columns=["Month","Value"]
# df.set_index('Month',inplace=True)

# Trend=findTrend(df)
# Seasonal=isSeasonal(df)
# Stationary=isStationary(df)
# findACF(df)



datafolder="E:\DataGenie-Hackathon\Dataset"
results=[]
i=0
new_data = pd.DataFrame(columns=['Date_type', 'Trend', 'Seasonal','Stationary','model'])

for filename in os.listdir(datafolder):
    df = pd.read_csv(os.path.join(datafolder, filename))

    df=df.drop(df.columns[0],axis=1)
    df['point_timestamp']=pd.to_datetime(df['point_timestamp'],infer_datetime_format=True)
    df.columns=["Date","Value"]
    df.set_index('Date',inplace=True)
    
    # print(df.isna().any())
    # if(df.isna().any()):
    #     df=missing_value(df)

    df=missing_value(df)

    print("************ Missing value ************",df.isna().sum()," File Upload ",filename)
    

    data_type=pd.infer_freq(df.index)


    Trend=findTrend(df)
    Seasonal=isSeasonal(df)
    Stationary=isStationary(df)

    if(Trend):
        model_type='ETS'
    elif(Seasonal):
        model_type='SARIMA'
    elif(Stationary):
        model_type='ARIMA'

    data=pd.DataFrame({'Date_type':data_type,'Trend':Trend,'Seasonal':Seasonal,'Stationary':Stationary,'model':model_type},index=[i])
    i=i+1
    new_data=new_data.append(data,ignore_index=True)

new_data.to_csv('Data.csv',index=False)
    



