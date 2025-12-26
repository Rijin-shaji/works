# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


# Load sample dataset
df=pd.read_csv("F:/house_price_data.csv")
X=df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']]
y=df['price']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeRegressor(criterion="poisson",  random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot the decision tree (limit depth for clarity)
plt.figure(figsize=(16, 8))
plot_tree(model, filled=True, feature_names=X.columns, max_depth=3)
plt.show()
