import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_null(data):
    null=data.isnull().sum()
    for index,value in null.items():
        if value!=0:
            new_data=data.interpolate()
        else:
            new_data=data

    return data
        



data=pd.read_csv("sample_9.csv")
data=data.drop(data.columns[0],axis=1)

data['point_timestamp']=pd.to_datetime(data['point_timestamp'],infer_datetime_format=True)
#indexdata=data.set_index(['point_timestamp'])

data=check_null(data)

