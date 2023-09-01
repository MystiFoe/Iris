import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

iris_data=pd.read_csv("D:\Machine Learning\Iris\Iris.csv")

X = iris_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris_data['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)

prediction=model.predict(X_test)

print(accuracy_score(y_test,prediction))

pickle.dump(model,open("iris_model.sav",'wb'))