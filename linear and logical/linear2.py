import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

df=pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'price':[550000,565000,610000,680000,725000]
})
x=df[['area']]
y=df['price']
# print(x)
# print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#print(y_pred)
print("MSE : ",mean_squared_error(y_test,y_pred))
print("R2 : ",r2_score(y_test,y_pred))
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='Blue')
plt.plot(x,model.predict(x),color="Red")
plt.title("LInear Regression - House Price ")
plt.show()
print(model.coef_)
print(model.intercept_)


