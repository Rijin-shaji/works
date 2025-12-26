import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("F:/marketing_campaign.csv", sep="\t")
df = df.dropna()

X= df[[ 'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue']]
y= df['Response']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler(with_mean=False)  
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_data = pd.DataFrame({
    'Year_Birth': [1980],
    'Income': [50000],
    'Kidhome': [1],
    'Teenhome': [0],
    'Recency': [10],
    'MntWines': [200],
    'MntFruits': [100],
    'MntMeatProducts': [150],
    'MntFishProducts': [80],
    'MntSweetProducts': [60],
    'MntGoldProds': [90],
    'NumDealsPurchases': [5],
    'NumWebPurchases': [10],
    'NumCatalogPurchases': [2],
    'NumStorePurchases': [3],
    'NumWebVisitsMonth': [8],
    'AcceptedCmp3': [1],
    'AcceptedCmp4': [0],
    'AcceptedCmp5': [1],
    'AcceptedCmp1': [0],
    'AcceptedCmp2': [1],
    'Complain': [0],
    'Z_CostContact': [3],
    'Z_Revenue': [500]
})

new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print("Prediction for new data:", prediction)