import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error

df=pd.DataFrame({
    'sleep':[4,5,7,8,9],
    'prd':[40,45,63,78,70]
})
#print(df)
plt.xlabel('sleep')
plt.ylabel('prd')
plt.scatter(df.sleep,df.prd,color='Red',marker='+')
#plt.show()
new_df=df.drop('prd',axis='columns')
#print(new_df)
prd=df.prd
#print(price)
reg=linear_model.LinearRegression()
reg.fit(new_df,prd)
print(reg.predict([[6]]))
print("Slope : ",reg.coef_)
print("Intercept :",reg.intercept_)
plt.xlabel('sleep')
plt.ylabel('prd')
plt.scatter(df.sleep,df.prd,color='Red',marker='+')
plt.plot(df.sleep,reg.predict(new_df))
plt.show()
mse= mean_squared_error(prd,reg.predict(new_df))
print("R2 : ",r2_score(prd,reg.predict(new_df)))
print(f"MSE : {mse}")
rmse=np.sqrt(mse)
print(f"RMSE : {rmse}")
mae=mean_absolute_error(prd,reg.predict(new_df))
print(f"MAE : {mae}")
mbe=np.mean(reg.predict(new_df)-prd)
print(f"MBE : {mbe}")