
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


data = {
  'Age': [22,25,28,35,40],
    'Income': [30,50,45,80,95],
    'cs': [650,720,680,750,800],
    'Approved':[0,1,0,1,1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Income', 'cs']]  
y = df['Approved']                        
     


model = LogisticRegression()
model.fit(X, y)


y_pred = model.predict(X)


df['Predicted'] = y_pred


print(df)
print("\nAccuracy:", model.score(X, y))


new_data = np.array([[28, 58,630]])
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
