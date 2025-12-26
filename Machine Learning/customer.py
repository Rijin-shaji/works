import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("F:/customer_spending_trends_v2_reindexed.csv")


le = LabelEncoder()
df['Customer_ID'] = le.fit_transform(df['Customer_ID'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Preferred_Product_Category'] = le.fit_transform(df['Preferred_Product_Category'])
df['Loyalty_Membership_Status'] = le.fit_transform(df['Loyalty_Membership_Status'])


X = df[[ 'index_no',	'Customer_ID',	'Age',	'Gender',	'City_Tier',	'Number_of_Visits_per_Month',	'Time_Spent_per_Visit',	'Average_Transaction_Value',	'Loyalty_Membership_Status',	'Purchase_Frequency',	'Preferred_Product_Category',]]
y = df[	'Spending_Trend']


vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100,random_state=42)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


