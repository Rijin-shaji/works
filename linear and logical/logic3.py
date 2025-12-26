import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score


x=[[2],[4],[6],[8],[9]]
y=[0,0,1,1,1]

plt.scatter(x,y,marker='+',color='red')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

print(X_test)

model=LogisticRegression()

model.fit(X_train,y_train)

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_train, model.predict(X_train), labels=[1,0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['age','broght_insurance'])
disp.plot(cmap='winter')

print(model.predict_proba(X_test))
print(model.score(X_test,y_test))

print(y_predicted)
print(model.coef_)
print(model.intercept_)

print(model.predict([[25]]))
print("Confusion Matrix:\n", cm)
acc = accuracy_score(y_train, model.predict(X_train))
prec = precision_score(y_train, model.predict(X_train), pos_label=1)
rec = recall_score(y_train, model.predict(X_train), pos_label=1)
f1 = f1_score(y_train, model.predict(X_train), pos_label=1)
print("Accuracy:", acc)
print("Precision (age):", prec)
print("Recall (age):", rec)
print("F1-score (age):", f1)
plt.show()