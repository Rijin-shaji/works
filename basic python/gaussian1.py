import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("F:/spam.csv")  


print(df.head())


df = df[['Category', 'Message']]


df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})




X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['Category'], test_size=0.2, random_state=42
)



vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()



model = GaussianNB()
model.fit(X_train_dense, y_train)

y_pred = model.predict(X_test_dense)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
