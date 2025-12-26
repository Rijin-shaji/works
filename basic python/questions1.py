import numpy as np
import pandas as pd

#1.
# a=np.random.randint(1,10,(3,3))
# print(a)
# det=np.linalg.det(a)
# print(det)
# inv=np.linalg.inv(a)
# print(inv)
# I=a*inv
# print(I)

#2.
# b=np.array([11,22,33,44,55,66,77,88,99,12,13,1,14,15,16,17,18,19,3,4,8,9,7,6])
# print(b)
# re=b.reshape(3,2,4)
# print(re)
# tr=np.transpose(re)
# print(tr)
# re1=np.shape(tr)
# print(re1)
# re2=np.shape(re)
# print(re2)

#3.
# a1=0
# a2=0
# a3=0
# a4=0
# a5=0
# a6=0
# d=np.random.randint(1,7,(1000))
# for i in d:
#     if i==1:
#         a1+=1
#     elif i==2:
#         a2+=1
#     elif i==3:
#         a3+=1
#     elif i==4:
#         a4+=1 
#     elif i==5:
#         a5+=1 
#     elif i==6:
#         a6+=1  
#     else:
#         continue
# print(f"1 - {a1}")  
# print(f"2 - {a2}")                       
# print(f"3 - {a3}")   
# print(f"4 - {a4}")  
# print(f"5 - {a5}")  
# print(f"6 - {a6}")  

#5.
df=pd.DataFrame({
    'Name':['Alice','Bob','Charlie','David','Evan'],
    'Age':[25,30,35,28,33],
    'Salary':[50000,6000,70000,55000,None],
    'City':['NY','LA','NY','LA','NY']
})

# print(df.head(2))
# print(df[df['City']=='NY'])
# print(df.groupby('City')['Salary'].mean())
# print(df['Salary'].mean())
# print(df.fillna(45250))

#6.

df1 = pd.read_csv("F:/titanic_dataset.csv")
# print(df1)
# print(df1.head(6))
# print(df1['Survived'].mean()*100)
#print(df1.value_counts('Pclass').max())
# print(df1[df1['Fare']>100])
# print(df1.fillna('Unknown'))
#print(df1.groupby('Fare')['Survived'].idxmax())
# print(pd.crosstab(df1['Sex'],df1['Survived']))
# print(pd.crosstab(df1['Pclass'],df1['Survived']))
#print(df1.groupby('Age')['Survived'].mean())
#print(df1.value_counts('Embarked'))
#print(df1.groupby('Survived')['Fare'].mean())
#**
# median_fare=df1['Fare'].median()
# high_fare=df1[df1['Fare']>median_fare]
# lower_fare=df1[df1['Fare']<=median_fare]
# hfar=high_fare['Survived'].mean()*100
# lfar=lower_fare['Survived'].mean()*100
# print(F"Higher Fare : {hfar}")
# print(F"Lower Fare : {lfar}")

#3.
#rolls=np.random.randint(1,7,size=1000)
#values,counts=np.unique(rolls,return_counts=True)
# for values,count in zip(values,counts):
#print(f"Dice face {values} : {counts} times")

#4.
# x=np.random.rand(5,2)
# y=np.random.rand(5,2)

# print(f"X coordinates : {x}")
# print(f"Y coordinates : {y}")
# distance=np.sqrt(np.sum((x-y)**2,axis=1))
# print(f"Euclidean distance between x  and y : {distance}")

