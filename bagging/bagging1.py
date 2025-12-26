import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# Load your dataset
# -----------------------------------
df = pd.read_csv("F:/New folder (2)/bankloan.csv")

# Split into features and target
X = df[['Age', 'Experience', 'Income', 'ZIP.Code', 'Family', 'CCAvg',
        'Education', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard']]
y = df['Personal.Loan']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# -----------------------------------
# Train-test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Bagging Classifier
# -----------------------------------
model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  
    n_estimators=100,                    
    max_samples=0.8,                     
    bootstrap=True,                      
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation
# -----------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
