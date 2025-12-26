import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df=pd.DataFrame({
    'area':[1,2,3,4,5],
    'price':[75,150,290,310,350]
})
x=df[['area']]
y=df['price']
# print(x)
# print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
#print(y_pred)
mse =mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mbe=np.mean(y_pred-y_test)
print("R2 : ",r2_score(y_test,y_pred))
print(f"MAE : {mae}")
print(f"MSE : {mse}")
print(f"RMSE : {rmse}")
print(f"MBE : {mbe}")
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='Blue')
plt.plot(x,model.predict(x),color="Red")
plt.title("LInear Regression - House Price ")
plt.show()
print("Slop :",model.coef_)
print("Intercept :",model.intercept_)
y_p=model.predict([[6]])
print(y_p)
