from flask import Flask, jsonify, request
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def index():
	return app.send_static_file('E:\DataGenie-Hackathon\frontend.html')

@app.route('/process_csv', methods=['POST'])
def process_csv():
	# Get the uploaded file from the request
	file = request.files['file']
	
	# Load the file into a pandas dataframe
	df = pd.read_csv(file)
	
	# Fit the ARIMA model
	model = ARIMA(df['y'], order=(1,1,1))
	results = model.fit()
	
	# Generate the forecast
	forecast = results.forecast(steps=10)
	
	# Compute the MAPE value
	actual = df['y'].values
	predicted = forecast.values
	mape = round((abs(actual - predicted) / actual).mean(), 4) * 100
	
	# Return the forecast and model details
	result = {
		'model': 'ARIMA',
		'mape': mape,
		'forecast': forecast.tolist()
	}
	return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True)
