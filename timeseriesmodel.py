import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller,kpss


def make_stationary(ts, lag=1):
    # Differencing to make the series stationary
    diff = ts.diff(lag).dropna()
    return diff

def make_non_stationary(ts, lag=1):
    # Cumulative sum to make the series non-stationary
    cumsum = ts.cumsum()
    return pd.Series([np.nan]*lag + cumsum.tolist(), index=ts.index)

def transform1(ts):
    # Apply log transformation to make the series more stationary
    return np.log(ts)

def retransform1(ts):
    # Apply exponentiation to revert back the log transformation
    return np.exp(ts)

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

