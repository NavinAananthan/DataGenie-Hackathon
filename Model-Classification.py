import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,kpss
import statsmodels.api as sm


def check_null(data):
    null=data.isnull().sum()
    for index,value in null.items():
        if value!=0:
            new_data=data.fillna(method='bfill')
        else:
            new_data=data

    return new_data
        

    
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

   
        

data=pd.read_csv("sample_9.csv")
data=data.drop(data.columns[0],axis=1)

data['point_timestamp']=pd.to_datetime(data['point_timestamp'],infer_datetime_format=True)
data=data.set_index(['point_timestamp'])

data=check_null(data)
#print(data)

print(check_seasonality(data))

#plt.plot(data)
#plt.show()
