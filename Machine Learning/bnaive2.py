import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv("F:/spam.csv")  
df = df[['Category', 'Message']]
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer(binary=True, stop_words='english')
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

model = BernoulliNB()
model.fit(X, y)

y_pred = model.predict(X)

print("Accuracy (Bernoulli NB):", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='macro'))
print("Recall:", recall_score(y, y_pred, average='macro'))
print("F1 Score:", f1_score(y, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
