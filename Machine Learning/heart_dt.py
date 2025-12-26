# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("F:/framingham.csv", encoding='latin-1')

X = df[['male','age','currentSmoker', 'cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes', 'totChol', 'sysBP','diaBP','BMI','heartRate', 'glucose']]  
y = df['TenYearCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(18, 10))
plot_tree(model, max_depth=3, filled=True, feature_names=X.columns, class_names=["No heart disease", "Heart disease"])
plt.show()