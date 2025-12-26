import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("F:/hospital_patients_dataset.csv")


le = LabelEncoder()
df['Age_Group'] = le.fit_transform(df['Age_Group'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Diagnosis_Type'] = le.fit_transform(df['Diagnosis_Type'])


X = df[[ 'Age_Group', 'Gender', 'Diagnosis_Type', 'Length_of_Stay', 'Lab_Test_Result', 'Previous_Admissions', 'Total_Billed_Amount']]
y = df['Readmitted_Within_30_Days']


vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



lr = LogisticRegression(max_iter=1000, random_state=42) 
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))


cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

new_data = pd.DataFrame({
    'Age_Group': [2],
    'Gender': [1],
    'Diagnosis_Type': [3],
    'Length_of_Stay': [5],
    'Lab_Test_Result': [7.8],
    'Previous_Admissions': [1],
    'Total_Billed_Amount': [1500]
})
new_pred = lr.predict(new_data)
print(f"Prediction for new data : {new_pred[0]}")









