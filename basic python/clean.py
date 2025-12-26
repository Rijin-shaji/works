#user input
#chech the string what's in it.
#print the integers only 
#remove the the other
t=[]

y=[x for x in input("Enter the word :").split()]
for i in y:
    try:
      z=int(i)
      print(z)
      t.append(z)  
    except:
      print('')
print(t)