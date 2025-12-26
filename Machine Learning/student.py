import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

df = pd.read_csv("F:/student-mat.csv", sep=";")
df = df.dropna()


y = (df['G3'] > 10).astype(int)


text_features = df['school'].astype(str) + " " + df['sex'].astype(str)

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(text_features)


numeric_cols = ['age','traveltime','studytime','failures','famrel','freetime',
                'goout','Dalc','Walc','health','absences','G1','G2']
X_numeric = df[numeric_cols].values


X = np.hstack([X_tfidf.toarray(), X_numeric])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


