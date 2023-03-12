import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb


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
    model = ExponentialSmoothing(df['Value'], trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)-1)

    plt.plot(df.index, df['value'], label='Actual')
    # plt.plot(.index, predictions, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')

    plt.show()


def arima(df):
    model = ARIMA(df['Value'], order=(1,1,1))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(df), end=len(df)-1, dynamic=False)

    plt.plot(df.index, df['value'], label='Actual')
    # plt.plot(.index, predictions, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')

    plt.show()


def xgboost(data):

    for i in range(1, 13):
        data[f'lag_{i}'] = data['value'].shift(i)

    train_data = data.iloc[:-20, :]
    test_data = data.iloc[-20:, :]

    X_train = train_data.drop('value', axis=1)
    y_train = train_data['value']

    X_test = test_data.drop('value', axis=1)
    y_test = test_data['value']

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, predictions, label='Predicted')
    plt.legend()
    plt.show()




def choose_model(model,df):

    if(model=='ETS'):
        ETS(df)
    elif(model=='SARIMA'):
        sarima(df)
    elif(model=='ARIMA'):
        arima(df)
    elif(model=='XGBOOST'):
        xgboost(df)

