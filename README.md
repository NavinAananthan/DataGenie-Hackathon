# DataGenie-Hackathon

```
Folder Structure
|__
  |_Model-Classification.py(Finding the best model for a given Time series dataset)
  |_Client.py(fetch data from database)
  |_frontend.html(Fronten page to upload CSV file)
 ```

* Model-Classification.py:-

     - This file takes a time series data as a input from the user and preprocess it such as filling null values, removing outliers from the dataset and extracting the features from the data such as extracting seasonality, trend, stationarity, Autocorrelation and other external factors which takes into consideration and finds the appropriate model based on the features extracted.
     - After extracting the features we run every time series dataset into it and create a CSV file accordingly for training it in a random forestclassifier to choose the best model accordingly
* Client.py:-

    - This process is to fetch the file from the user and pass the dataframe/CSV file into the model-classification which is generated as a pickle file and pass the dataframe into it and find the appropriate model and forecast the future based on the time series data
    - The user sends API request for a particular month to generate the predicted particular value
* Frontend.html

    - This Is to create a front end page to upload a CSV file and plot the forecast for the given dataset
 
