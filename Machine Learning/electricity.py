import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("F:/substation_data.csv")

X = df[[ 'Power_Load_kWh', 'Temperature_C', 'Humidity_percent', 'Wind_Speed_m_s', 'Transformer_Age_years', 'Maintenance_Freq_per_year', 'Power_Factor']]
y = df['Failure_Event']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


new_data = pd.DataFrame({
    'Power_Load_kWh': [5000],
    'Temperature_C': [35],
    'Humidity_percent': [70],
    'Wind_Speed_m_s': [5],
    'Transformer_Age_years': [10],
    'Maintenance_Freq_per_year': [4],
    'Power_Factor': [0.95]
})
new_data_scaled = scaler.transform(new_data)
new_prediction = nb.predict(new_data_scaled)
print(f"Prediction for new data: {new_prediction[0]}")