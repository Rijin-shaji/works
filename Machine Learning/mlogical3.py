
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


data = {
      'Hours': [2,4,6,8,9],
    'Attendance': [60,80,75,90,88],
    'Passed':      [0,0,1,1,1]
}

df = pd.DataFrame(data)


X = df[['Hours', 'Attendance']]   
y = df['Passed']                               


model = LogisticRegression()
model.fit(X, y)


y_pred = model.predict(X)


df['Predicted'] = y_pred


# print(df)

new_data = np.array([[12, 80]])  
prediction = model.predict(new_data)
print("\nPredicted (1=Pass, 0=Fail):", prediction[0])
print("\nPrediction Probabilities:\n", model.predict_proba(new_data))
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print(model.predict_proba(X))
print(model.score(X,y))
prec = precision_score(y, model.predict(X), pos_label=1)
rec = recall_score(y, model.predict(X), pos_label=1)
f1 = f1_score(y, model.predict(X), pos_label=1)
print("Precision (age):", prec)
print("Recall (age):", rec)
print("F1-score (age):", f1)
print("\nConfusion Matrix:\n", confusion_matrix(y, model.predict(X)))
print("\nAccuracy:", model.score(X, y))
