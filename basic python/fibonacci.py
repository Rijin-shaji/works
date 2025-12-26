#user input ,fibonacci limit.
#create the fibonacci numbers up to the limit 
#print 

p=0
add=1
n=int(input("Enter the number : "))
for i in range(n):
  k=p+add
  print(p,end=" ")
  p=add
  add=k

  


