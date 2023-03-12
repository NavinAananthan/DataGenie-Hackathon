from flask import Flask, request, render_template
from classifier import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from DatasetGeneration import *
from classifier import *


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file from the request
    file = request.files['csvfile']
    
    df = pd.read_csv(file)

    #print(df.head())

    data=preprocess(df)

    datatype=find_datatype(data)
    seasonlaity=isSeasonal(data)
    stationaryity=isStationary(data)
    trend=findTrend(data)

    

    return "Sucessfully Uploaded"
    
    
if __name__ == '__main__':
    app.run(debug=True)
