import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("F:/Movie_Dataset_500.csv")

le = LabelEncoder()
df['Movie_ID'] = le.fit_transform(df['Movie_ID'])

X = df[[ 'Movie_ID', 'Budget (in M$)', 'Runtime (min)', 'Screens Released', 'Social Media Engagement', 'Star Cast Popularity', 'Critics Rating', 'Viewer Rating']]
y = df['Box Office Category']

vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


new_data = pd.DataFrame({
    'Movie_ID': [501],
    'Budget (in M$)': [100],
    'Runtime (min)': [120],
    'Screens Released': [3000],
    'Social Media Engagement': [50000],
    'Star Cast Popularity': [8.5],
    'Critics Rating': [7.2],
    'Viewer Rating': [8.0]
})
new_data_scaled = scaler.transform(new_data)
new_prediction = nb.predict(new_data_scaled)
print(f"Prediction for new data: {new_prediction[0]}")