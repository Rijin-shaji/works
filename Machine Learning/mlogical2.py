
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score


data = {
    'Age': [22,25,28,35,40],
    'Income': [30,50,45,80,95],
    'cs': [650,720,680,750,800],
    'Approved':[0,1,0,1,1]
}

df = pd.DataFrame(data)


X = df[['Age', 'Income', 'cs']]  
y = df['Approved']                       


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)



print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


new_data = np.array([[28, 58,630]])  
prediction = model.predict(new_data)
print("\nPredicted (1=Pass, 0=Fail):", prediction[0])
print(model.predict_proba(X_test))
print(model.score(X_test,y_test))

print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Model Probabilities:\n", model.predict_proba(X_test))
print("Model Score:", model.score(X_test,y_test))
print("\nAccuracy:", model.score(X, y))
prec = precision_score(y_train, model.predict(X_train), pos_label=1)
rec = recall_score(y_train, model.predict(X_train), pos_label=1)
f1 = f1_score(y_train, model.predict(X_train), pos_label=1)
print("Precision (age):", prec)
print("Recall (age):", rec)
print("F1-score (age):", f1)
    
