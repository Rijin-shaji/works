import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("F:/New folder (2)/bankloan.csv")


features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
            'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard']
target = 'Personal.Loan'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


new_data = pd.DataFrame({
    'Age': [30],
    'Experience': [5],
    'Income': [60000],
    'Family': [2],
    'CCAvg': [1.5],
    'Education': [1],
    'Mortgage': [0],
    'Securities.Account': [0],
    'CD.Account': [0],
    'Online': [1],
    'CreditCard': [1]
})

new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print("Prediction for new data:", prediction)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
