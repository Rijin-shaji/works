n=input("Enter the word : ")
d={}
for i in n:
 if i in d:
  d[i]=d[i]+1
 else:
  d[i]=1
print(d) 