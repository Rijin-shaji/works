import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("F:/insurance_data.csv", encoding='latin-1')

X = df[[ 'age']]
y = df['bought_insurance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# plt.figure(figsize=(8,6))
# plt.scatter(X_test['age'], X_test['bought_insurance'], c=y_pred)
# plt.title("KNN Classification Result")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.colorbar(label="Predicted Class")

# plt.show()
