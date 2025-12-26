import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("F:/framingham.csv")

#cleaning data
df = df.dropna()
print(df.head())

X = df[['male','age','currentSmoker', 'cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes', 'totChol', 'sysBP','diaBP','BMI','heartRate', 'glucose']]  
y = df['TenYearCHD']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


new_data = pd.DataFrame({
    'male': [1],
    'age': [45],
    'currentSmoker': [1],
    'cigsPerDay': [30],
    'BPMeds': [1],
    'prevalentStroke': [0],
    'prevalentHyp': [1],
    'diabetes': [0],
    'totChol': [250],
    'sysBP': [140],
    'diaBP': [90],
    'BMI': [28],
    'heartRate': [80],
    'glucose': [85]
})

new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print("Prediction for new data:", prediction)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

