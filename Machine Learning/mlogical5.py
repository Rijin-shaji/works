
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


df = pd.read_csv("F:/titanic_dataset.csv")

df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
X = df[['Pclass', 'Age', 'Fare', 'Sex']]  
y = df['Survived']                        


model = LogisticRegression()
model.fit(X, y)


y_pred = model.predict(X)


df['Predicted'] = y_pred


#print(df)
new_data = np.array([[1, 34, 120.25, 0]])
prediction = model.predict(new_data)
print("\nPredicted (1=Pass, 0=Fail):", prediction[0])
print("\nPrediction Probabilities:\n", model.predict_proba(new_data))
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