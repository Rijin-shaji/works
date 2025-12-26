import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

df=pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'price':[550000,565000,610000,680000,725000]
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
# print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='Red',marker='+')
plt.plot(df.area,reg.predict(new_df))
#plt.show()
print("MSE : ",mean_squared_error(price,reg.predict(new_df)))
print("R2 : ",r2_score(price,reg.predict(new_df)))
