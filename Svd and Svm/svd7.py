import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# 1. Load Your Dataset
df = pd.read_csv("F:/marketing_campaign.csv", sep="\t")
df = df.dropna()

X= df[[ 'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue']]
y= df['Response']

# 3. Scale (recommended for SVD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply Truncated SVD
svd = TruncatedSVD(n_components=5, random_state=42) 
X_reduced = svd.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_reduced.shape)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.25, random_state=42
)

# 6. Train a Classifier
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# 7. Predictions & Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", acc)
print("\nConfusion Matrix:\n", cm)