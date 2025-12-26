import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv("F:/car_data.csv")
df = pd.DataFrame(data)

x=df[['Engine_Size (L)','Weight (kg)','Horsepower']]
y=df['MPG (Miles_per_Gallon)']

scalar=StandardScaler()
x_scaled =scalar.fit_transform(x)
y_scaled =(y-y.mean())/y.std()

model=LinearRegression()
model.fit(x_scaled,y_scaled)

coef=pd.Series(model.coef_,index=x.columns)
print(coef)
