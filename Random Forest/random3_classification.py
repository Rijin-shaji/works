# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("F:/Pumpkin_Seeds_Dataset.xlsx")

X = df[[ 'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity',
        'Extent', 'Roundness', 'Aspect_Ration', 'Compactness']]
y = df['Class']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf =RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=4,min_samples_leaf=2,max_features='sqrt',bootstrap=True,random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()