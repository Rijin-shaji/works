from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load dataset
df = pd.read_csv("F:/New folder (2)/bankloan.csv")
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
            'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard']
target = ['Personal.Loan']

X = df[features]
y = df[target]

# 2) Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Apply PCA

pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("After PCA:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 4) Train/Test split

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# 5) Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# 6) Make predictions
y_pred = model.predict(X_test)

# 7) Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))