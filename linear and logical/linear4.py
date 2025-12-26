import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error

df=pd.DataFrame({
    'area':[1,2,3,4,5],
    'price':[75,150,290,310,350]
})
#print(df)
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='Red',marker='+')
#plt.show()
new_df=df.drop('price',axis='columns')
#print(new_df)
price=df.price
#print(price)
reg=linear_model.LinearRegression()
reg.fit(new_df,price)
print(reg.predict([[6]]))
print("Slope : ",reg.coef_)
print("Intercept :",reg.intercept_)
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='Red',marker='+')
plt.plot(df.area,reg.predict(new_df))
plt.show()
mse= mean_squared_error(price,reg.predict(new_df))
print("R2 : ",r2_score(price,reg.predict(new_df)))
print(f"MSE : {mse}")
rmse=np.sqrt(mse)
print(f"RMSE : {rmse}")
mae=mean_absolute_error(price,reg.predict(new_df))
print(f"MAE : {mae}")
mbe=np.mean(reg.predict(new_df)-price)
print(f"MBE : {mbe}")