import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("F:/spam.csv", encoding='latin-1')


X = df['Message']
y = df['Category']


vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

le = LabelEncoder()
y_pred_numeric = le.fit_transform(y_pred)
y_test_numeric = le.transform(y_test) 

# plt.figure(figsize=(8,6))
# plt.scatter(X_test['Area'], X_test['Perimeter'],c=y_pred_numeric, 
#     cmap='viridis',
#     alpha=0.7)
# plt.title("KNN Classification Result")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.colorbar(label="Predicted Class")

# plt.show()