import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima(df):
    model = SARIMAX(df['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit()
    predictions = result.predict(start=len(df), end=len(df)-1, dynamic=False)

    plt.plot(df.index, df['value'], label='Actual')
    # plt.plot(.index, predictions, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')

    plt.show()



def ETS(df):
    return True

def arima(df):
    return True

def xgboost(df):
    return True

def probhet(df):
    return True

def choose_model(model,df):

    if(model=='ETS'):
        ETS(df)
    elif(model=='SARIMA'):
        sarima(df)
    elif(model=='ARIMA'):
        arima(df)
    elif(model=='XGBOOST'):
        xgboost(df)
    elif(model=='Probhet'):
        probhet(df)

