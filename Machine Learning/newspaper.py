import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_excel("F:/Uci-newssport.xlsx")
df = df.dropna()

X = df['NEWS']
y = df['CLASS']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


new_text = ["The team secured a thrilling victory in the championship game."]
new_vector = vectorizer.transform(new_text)
prediction = model.predict(new_vector)
print("Prediction for new data:", prediction)

