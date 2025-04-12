# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: TRISHA PRIYADARSHNI PARIDA
RegisterNumber:  212224230293
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:


<img width="823" alt="image" src="https://github.com/user-attachments/assets/d752f293-15f7-4ca0-853e-4b45819b4ff1" />

<img width="833" alt="image" src="https://github.com/user-attachments/assets/3c912d3f-401e-4708-a714-a42a242c4cad" />

<img width="733" alt="image" src="https://github.com/user-attachments/assets/2000f38f-c5f1-4260-87ce-9f3f0f445306" />

<img width="683" alt="image" src="https://github.com/user-attachments/assets/0e63f436-5ec7-4248-8dd3-d96d25fde16b" />

<img width="635" alt="image" src="https://github.com/user-attachments/assets/7cad435f-5efe-4b5c-aac4-aa7a49c525e7" />

<img width="121" alt="image" src="https://github.com/user-attachments/assets/ec1430e6-0b90-41df-99f1-360d5e46fe3b" />

<img width="350" alt="image" src="https://github.com/user-attachments/assets/82cf2ef0-3db2-4139-bad3-d88b0e14f7d1" />

<img width="452" alt="image" src="https://github.com/user-attachments/assets/668f4a00-2a44-4af3-b261-48280325d9c6" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
