import pandas as pd
import numpy as np
x=pd.Series(np.random.randint(10,60,10))
#print(x)
# print(x.head())
# print(x.tail())
# print(x.shape)
# print(x.size)
df=pd.DataFrame({
    'name':['Alice','Bob','Charlie','David','Roger' ],
    'age':[25,30,35,40,28],
    'city':['Ny','LA','SF','LA','NA'],
    'salary':[200,300,400,500,600]})
df1=pd.DataFrame({
    'name':['A','B','C'],
    'age':[25,30,35],
    'place':['Ny','LA','SF'],
    'salary':[200,300,400]})
df2=pd.DataFrame({
    'date':['2025-06-19','2025-05-20','2026-01-01','2026-01-05','2023-06-19','2022-05-20','2010-01-01','2000-01-05'],
    'time':['10:30:20','11:50:10','1:00:30','11:30:20','19:50:10','10:00:30','1:50:30','15:30:20'],
    'number':[1,2,3,4,5,6,7,8]})
# print(df)
# print(df.head())
# print(df.tail())
# print(df.describe())
# print(df.info())
# print(df.shape)
# print(df.columns)
# print(df.index)
# print(df.loc[0])
# print(df.loc[:,'name'])
#print(df.iloc[0,0])
#df['age']=['a','b','c','d','e']
# df['country']='USA'
#df.drop(label=None,axis=0,index=None,columns=None,inplace=false)
# print(df.isna())
# print(df.dropna())
# print(df.fillna(0))
# print(df.replace(None,'B'))
#print(df.sort_values('age',ascending=True))
#print(df.sort_index())
#print(df.sort_index(ascending=False))
# print(df.rank(axis=0))
# print(df[df['age']>20])
# print((df['age']>20)&(df['city']=='NY'))
#print(df.query('age>10 and age<50'))
# print(df.groupby('city')['age'].mean())
# print(df.groupby('city')['age'].agg('mean'))
# print(df.groupby('city')['age'].agg(['mean,'min']))
#print(pd.crosstab(df1['name'],df1['salary']))
#print(df1.apply(lambda))
# x=pd.concat([df,df1])
# print(pd.merge(x,df1))
#print(df.groupby('name')['age'].mean())
#print(pd.cut(df['age'],bins=[0,18,35,60],labels=['Youth','Adults','Seniors']))
#print(df.join(df2))
# print(pd.to_datetime(df2['date']+ '  ' +df2['time']))
# print(pd.date_range(start='2025-06-19',end='2028-01-01',freq='YE'))     #freq M for months , without freq for dates , freq Y for year
df2['date']=pd.to_datetime(df2['date'])
df2['time']=pd.to_datetime(df2['time'])
# print(df2['date'].dt.year)  # dt.year for showing year, for month and stae we can change it to month and date.
df2.set_index('date',inplace=True)
# print(df2.resample('YE', on='date').max())
# print(df2.resample('YE', on='date').sum())
# print(df2.shift(periods=3,fill_value=0))
# e=df2.tz_localize('UTC')
# print(e.tz_convert('Asia/Kolkata'))
#print(df['name'].str.upper())
#print(df['name'].str.lower())
#print(df['name'].str.strip())
# print(df['name'].str.replace('Alice','Abin'))
#print(df)
# df['name']=df['name'].replace(to_replace=[None], value='Roger')
# print(df)
#print(df['name'].str.contains('Bob'))
#print(df['name'].str.startswith('R'))
#print(df['name'].str.split())
#print(df['name'].str.get(-1))
#print(df['name'].str.join('$'))
#print(df['name'].str.extract())
#print(df['name'].str.findall(''))