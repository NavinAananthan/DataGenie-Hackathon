import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def predict_model(datatype,seasonal,stationarity,trend):

    model=clf.predict(datatype,seasonal,stationarity,trend)

    return model


data=pd.read_csv('Data.csv')

le=LabelEncoder()

data["Date_type"]=le.fit_transform(data["Date_type"])
data["Trend"]=le.fit_transform(data["Trend"])
data["Seasonal"]=le.fit_transform(data["Seasonal"])
data["Stationary"]=le.fit_transform(data["Stationary"])
data["model"]=le.fit_transform(data["model"])

x=data.drop("model",axis=1)
y=data["model"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
