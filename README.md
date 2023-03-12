# DataGenie-Hackathon

```
Folder Structure
|__
  |_DatasetGeneration.py(Creates the best time series model dataset)
  |_classifier.py(Classifies the time series)
  |_timeseriesmodel.py(choose the appropriate model)
  |_app.py(Server side to get dataset)
  |_frontend.html(Fronten page to upload CSV file)
  |_Data.csv(dataset for best model and features)
 ```

* DatasetGeneration.py:-

     - The process of creating a new dataset is based on extracting the features from the time series dataset for hourly, weekly, monthly etc and check if their seasonal stationary trend and other set of features and create a new dataset based on the features
     - These features are add into a new Data.csv file based on the extracted features we find which model to use and put it in the target column for eg it is seasonal we put SARIMA and seasonal column is set true and other columns are set to false and even there is a column for monthly, weekly, daily or hourly data.
* Classifier.py:-

    - The Data.csv file is then label encode based on the features for true false it is set to 1 and 0 and for target variables it is encoded and we split the dataset into training and testing dataset
    - Then we pass the training dataset into XGBoost classifier model to train our best time series model and then we predic the test data set.
* app.py

    - This is a backend to connect with frontend here the user uploads the dataset and we get it here and here we preprocess the file and finc the best time series model by importing the classifier file and find the best model.
    - Then we pass the best model to timeseriesmodel.py which we have imported it in this file.
* timeseriesmodel.py

   - Here we get the best model and accordingly we then forecast and predict the time series data by passing the time series dataset.
   - Here i have assumed the parameter for the models and have given as it is mentioned we should not use Auto arima to find the best prarmeters
* frontend.html

   - Here we create a front end pasge using html and style it with css and use javascript functions to get the file from the user and send it to the server.
   - Error occured while plotting the forecast in UI part
   - I have attached below the screen shot of it.
 
 
![image](https://user-images.githubusercontent.com/81963819/224550095-e9595c27-5f4e-485b-b3e7-c30f4752648e.png)


 * problems occured :-
 
     - The first problem occured is it is not forecasting correctly for the given sample time series dataset but working for other online time series dataset
     - Didnt have time to implement the date functionality to find the predicted value for particular date
     - Error while plotting in the UI part 
  
 * Note: There are many features to take in consideration for the time series dataset but i have choosen only limited. 
