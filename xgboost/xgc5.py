# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load dataset


df = pd.read_csv("F:/data.csv", encoding='latin-1')

X = df[[ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
        'symmetry_mean', 'fractal_dimension_mean']]
y = df['diagnosis']

y = y.map({'M': 1, 'B': 0})


# Encode labels (spam/ham -> 1/0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Convert text into numeric features using TF-IDF

# vectorizer = TfidfVectorizer(stop_words='english')
# X_vect = vectorizer.fit_transform(X)

# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded , test_size=0.2, random_state=42
)

# Train XGBoost Classifier

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # for binary classification
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Make predictions

y_pred = xgb_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))