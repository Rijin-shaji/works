from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
x=[[2],[4],[6],[8],[9]]
y=[0,0,1,1,1]

model.fit(x,y)

x_test=[[2],[4],[6],[8],[9]]
y_predicted=model.predict(x_test)

print(y_predicted)