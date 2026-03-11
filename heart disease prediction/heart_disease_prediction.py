import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv data to pd dataframe
heart_data = pd.read_csv(r"F:\heart_disease_data.csv")

#print first 5 rows of dataset

print(heart_data.head())

#print last 5 rows of dataset

print(heart_data.tail())

#number of rows and columns in dataset
print(heart_data.shape)

#getting info about the dataset
print(heart_data.info())

#checking for missing values
print(heart_data.isnull().sum())

#statistical measures
print(heart_data.describe())

#checking ditribution of target values
print(heart_data['target'].value_counts())
 
# 1 -> defective heart and 0 -> healthy heart
#splitting features and target 
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)

#splitting training and test data
X_train, X_test, Y_train, Y_test= train_test_split( X, Y, train_size=0.2, stratify=Y, random_state=2 )
print (X.shape, X_train.shape, X_test.shape)

#training model
model = LogisticRegression()

#traing model with traing data

model.fit(X_train,Y_train)

#model Evaluation
X_train_prediction = model.predict(X_train)
traing_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy of training data : ",traing_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy of testing data : ",testing_data_accuracy)


input_data = (44,1,1,120,263,0,1,173,0,0,2,0,3)
input_data_df= pd.DataFrame([input_data],columns=X.columns)

prediction= model.predict(input_data_df)
print(prediction)

if prediction[0] == '0':
    print("The person does not has a heart disease")
else:
    print("The person has a heart disease")