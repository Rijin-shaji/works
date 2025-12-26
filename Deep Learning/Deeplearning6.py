import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
df = pd.read_csv("F:\Walmart.csv")

x = df[["Store","Weekly_Sales","Temperature","Fuel_Price","CPI","Holiday_Flag"]]
y = df["Unemployment"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)

mse =mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mbe=np.mean(y_pred-y_test)
print("R2 : ",r2_score(y_test,y_pred))
print(f"MSE : {mse}")

