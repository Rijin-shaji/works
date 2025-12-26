from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("F:/spam.csv")  
df = df[['Category', 'Message']]


df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message']).toarray()
y = df['Category']


model = GaussianNB()
model.fit(X, y)


y_pred = model.predict(X)

print("Accuracy (Gaussian NB):", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='macro'))
print("Recall:", recall_score(y, y_pred, average='macro'))
print("F1 Score:", f1_score(y, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
