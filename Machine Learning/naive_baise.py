# # 1) Gaussian Naive Bayes (For Numeric Data)
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix

# data = load_iris()
# X = data.data
# y = data.target
# print(data)
# model = GaussianNB()
# model.fit(X, y)

# y_pred = model.predict(X)

# print("Accuracy (Gaussian NB):", accuracy_score(y, y_pred))
# print("Precision:", precision_score(y, y_pred, average='macro'))
# print("Recall:", recall_score(y, y_pred, average='macro'))
# print("F1 Score:", f1_score(y, y_pred, average='macro'))
# print("Confusion Matrix:\n", confusion_matrix(y, y_pred))




# # #train test
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

# data = load_iris()
# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = GaussianNB()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy (Gaussian NB):", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average='macro'))
# print("Recall:", recall_score(y_test, y_pred, average='macro'))
# print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))













# # 2) Multinomial Naive Bayes (For text data / word counts)
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix

# # Example text data
# texts = ["I love apples",
#          "I hate apples",
#          "I love oranges",
#          "I dislike bananas"]

# labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# # Convert text → word count matrix
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(texts)
# print(X)
# # Model
# model = MultinomialNB()
# model.fit(X, labels)

# # Predict on same data
# y_pred = model.predict(X)

# print("Accuracy (Multinomial NB):", accuracy_score(labels, y_pred))
# print("Precision:", precision_score(labels, y_pred, average='macro'))
# print("Recall:", recall_score(labels, y_pred, average='macro'))
# print("F1 Score:", f1_score(labels, y_pred, average='macro'))
# print("Confusion Matrix:\n", confusion_matrix(labels, y_pred))



# # # #train test
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix

# # Example text documents
# texts = ["I love apples",
#          "I hate apples",
#          "I love oranges",
#          "I dislike bananas"]

# # Labels (1 = positive, 0 = negative)
# labels = [1, 0, 1, 0]

# # Convert text to word count matrix
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(texts)

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# # Model
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Accuracy
# print("Accuracy (Multinomial NB):", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average='macro',zero_division=1))
# print("Recall:", recall_score(y_test, y_pred, average='macro',zero_division=1))
# print("F1 Score:", f1_score(y_test, y_pred, average='macro',zero_division=1))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))













# #  # 3) Bernoulli Naive Bayes (For Binary Presence/Absence)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix

# Same example text
texts = ["I love apples",
         "I hate apples",
         "I love oranges",
         "I dislike bananas"]

labels = [1, 0, 1, 0]

# Convert text to binary presence (1) or absence (0)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Model
model = BernoulliNB()
model.fit(X, labels)

# Predict on same data
y_pred = model.predict(X)

print("Accuracy (Bernoulli NB):", accuracy_score(labels, y_pred))
print("Precision:", precision_score(labels, y_pred, average='macro'))
print("Recall:", recall_score(labels, y_pred, average='macro'))
print("F1 Score:", f1_score(labels, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(labels, y_pred))



#train test
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix

# Same example text
texts = ["I love apples",
         "I hate apples",
         "I love oranges",
         "I dislike bananas"]

labels = [1, 0, 1, 0]

# Convert text → binary presence (1) or absence (0)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Model
model = BernoulliNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy (Bernoulli NB):", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro',zero_division=1))
print("Recall:", recall_score(y_test, y_pred, average='macro',zero_division=1))
print("F1 Score:", f1_score(y_test, y_pred, average='macro',zero_division=1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))