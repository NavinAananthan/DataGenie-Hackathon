from flask import Flask, request, render_template
from classifier import *
import pandas as pd
from DatasetGeneration import *
from classifier import *
from timeseriesmodel import *


def encoder(flag):
    if(flag):
        return 1
    else:
        return 0


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('frontend.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file from the request
    file = request.files['csvfile']
    
    df = pd.read_csv(file)

    data=preprocess(df)

    datatype=find_datatype(data)
    seasonlaity=isSeasonal(data)
    stationaryity=isStationary(data)
    trend=findTrend(data)

    seasonlaity=encoder(seasonlaity)
    stationaryity=encoder(stationaryity)
    trend=encoder(trend)

    model=predict_model(datatype,seasonlaity,stationaryity,trend)

    choose_model(model)

    return "Sucessfully Uploaded"
    
    

if __name__ == '__main__':
    app.run(debug=True)
